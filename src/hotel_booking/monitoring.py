"""Prometheus metrics of the prediction API (contract C10 in `docs/source/contracts.md`)."""

import time
from collections.abc import Awaitable, Callable, Sequence

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from starlette.routing import Match

from hotel_booking.schemas import InputWarning

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
PROBABILITY_BUCKETS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
UNMATCHED_ROUTE = "unmatched"


def error_type(status: int) -> str | None:
    """Classify a response status as an error type.

    Args:
        status: HTTP status code.

    Returns:
        `validation` for 422, `unavailable` for 503, `client` for other 4xx, `internal` for other 5xx, else `None`.
    """
    if status == 422:
        return "validation"
    if status == 503:
        return "unavailable"
    if 400 <= status < 500:
        return "client"
    if status >= 500:
        return "internal"
    return None


class ApiMetrics:
    """The metrics of one API instance, in their own registry so several apps can live in one process."""

    def __init__(self) -> None:
        """Create the registry and the metrics."""
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "api_requests", "HTTP requests by route and status code.", ["endpoint", "status"], registry=self.registry
        )
        self.latency = Histogram(
            "api_request_latency_seconds",
            "Time to answer a request, by route.",
            ["endpoint"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.errors = Counter(
            "api_errors", "Failed requests by route and error type.", ["endpoint", "type"], registry=self.registry
        )
        self.probability = Histogram(
            "api_prediction_probability",
            "Predicted cancellation probabilities, one observation per booking.",
            buckets=PROBABILITY_BUCKETS,
            registry=self.registry,
        )
        self.predictions = Counter(
            "api_predictions", "Scored bookings by decision.", ["outcome"], registry=self.registry
        )
        self.input_warnings = Counter(
            "api_input_warnings",
            "Bookings with a value unseen in train, by column and issue.",
            ["column", "issue"],
            registry=self.registry,
        )
        self.model_info = Gauge(
            "api_model_info",
            "The served model: 1 for the loaded version.",
            ["model_version", "backend"],
            registry=self.registry,
        )

    def set_model(self, version: str | None, backend: str = "joblib") -> None:
        """Mark which model is served; clears the previous one.

        Args:
            version: The served model version, or `None` when no model is loaded.
            backend: Inference backend.
        """
        self.model_info.clear()
        if version is not None:
            self.model_info.labels(model_version=version, backend=backend).set(1)

    def observe_predictions(self, probabilities: Sequence[float], threshold: float) -> None:
        """Record the probabilities and decisions of scored bookings.

        Args:
            probabilities: One probability per booking.
            threshold: Decision threshold.
        """
        canceled = 0
        for value in probabilities:
            self.probability.observe(value)
            canceled += value >= threshold
        self.predictions.labels(outcome="canceled").inc(canceled)
        self.predictions.labels(outcome="not_canceled").inc(len(probabilities) - canceled)

    def observe_warnings(self, warnings: Sequence[InputWarning]) -> None:
        """Count the bookings with unseen values.

        Args:
            warnings: Warnings returned with a prediction.
        """
        for warning in warnings:
            self.input_warnings.labels(column=warning.column, issue=warning.issue).inc(warning.rows)


def route_label(app: FastAPI, request: Request) -> str:
    """The route template of a request, so path parameters never create new label values.

    Args:
        app: The application.
        request: The request.

    Returns:
        The route path, or `unmatched` for unknown paths.
    """
    for route in app.router.routes:
        match, _ = route.matches(request.scope)
        if match == Match.FULL:
            return str(getattr(route, "path", UNMATCHED_ROUTE))
    return UNMATCHED_ROUTE


def instrument(app: FastAPI, metrics: ApiMetrics) -> None:
    """Add request metrics and the `GET /metrics` endpoint to an app.

    Args:
        app: The application.
        metrics: Its metrics.
    """

    @app.middleware("http")
    async def record_request(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        endpoint = route_label(app, request)
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            metrics.latency.labels(endpoint=endpoint).observe(time.perf_counter() - start)
            metrics.requests.labels(endpoint=endpoint, status=str(status)).inc()
            kind = error_type(status)
            if kind:
                metrics.errors.labels(endpoint=endpoint, type=kind).inc()

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics() -> Response:
        """Expose the metrics in the Prometheus text format."""
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)
