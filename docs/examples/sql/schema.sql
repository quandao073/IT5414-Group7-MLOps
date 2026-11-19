-- Prediction log and delayed labels (contract C9 in docs/source/contracts.md).
-- Idempotent: the API runs it at startup, so it is safe to run again.

-- One row per scored booking; a batch request writes one row per booking with the same request_id.
CREATE TABLE IF NOT EXISTS prediction_logs (
    id            BIGSERIAL        PRIMARY KEY,
    request_id    UUID             NOT NULL,
    row_index     INTEGER          NOT NULL DEFAULT 0,
    created_at    TIMESTAMPTZ      NOT NULL DEFAULT now(),
    endpoint      TEXT             NOT NULL,
    model_version TEXT             NOT NULL,
    backend       TEXT             NOT NULL DEFAULT 'joblib',
    booking_id    BIGINT,
    replay_month  TEXT,
    input         JSONB            NOT NULL,
    probability   DOUBLE PRECISION NOT NULL CHECK (probability BETWEEN 0 AND 1),
    threshold     DOUBLE PRECISION NOT NULL,
    is_canceled   BOOLEAN          NOT NULL,
    latency_ms    DOUBLE PRECISION NOT NULL,
    warnings      JSONB            NOT NULL DEFAULT '[]'::jsonb
);

CREATE INDEX IF NOT EXISTS prediction_logs_created_at_idx ON prediction_logs (created_at);
CREATE INDEX IF NOT EXISTS prediction_logs_replay_month_idx ON prediction_logs (replay_month);
CREATE INDEX IF NOT EXISTS prediction_logs_model_version_idx ON prediction_logs (model_version);
CREATE INDEX IF NOT EXISTS prediction_logs_booking_id_idx ON prediction_logs (booking_id);

-- Labels released by the replay after a whole month has been scored.
CREATE TABLE IF NOT EXISTS labels (
    booking_id  BIGINT      PRIMARY KEY,
    is_canceled SMALLINT    NOT NULL CHECK (is_canceled IN (0, 1)),
    month       TEXT        NOT NULL,
    released_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS labels_month_idx ON labels (month);
