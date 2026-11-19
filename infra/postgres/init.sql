-- Runs once, when the postgres volume is created. One database per consumer.
CREATE DATABASE mlflow;         -- MLflow tracking and Model Registry
CREATE DATABASE airflow;        -- Airflow metadata
CREATE DATABASE hotel_booking;  -- prediction log and delayed labels (docs/examples/sql/schema.sql)
