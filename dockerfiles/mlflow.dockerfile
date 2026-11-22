# Self-hosted MLflow tracking server and Model Registry (PostgreSQL backend, artifacts on SeaweedFS S3).
# The official image lacks the PostgreSQL driver and boto3.
FROM ghcr.io/mlflow/mlflow:v3.18.0

RUN pip install --no-cache-dir psycopg2-binary==2.9.13 boto3==1.44.6
