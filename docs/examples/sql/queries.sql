-- Example queries on the prediction log (contract C9 in docs/source/contracts.md).
-- Parameters are written as :name, e.g. :month = '2017-03'.

-- 1. Drift: model inputs of one replayed month, one row per booking, as columns.
--    Compare with the reference data (data/processed/train.csv) in the drift service.
SELECT p.booking_id, p.created_at, p.probability, i.*
FROM prediction_logs AS p
CROSS JOIN LATERAL jsonb_to_record(p.input) AS i(
    hotel TEXT, arrival_date_month TEXT, meal TEXT, market_segment TEXT, distribution_channel TEXT,
    reserved_room_type TEXT, deposit_type TEXT, customer_type TEXT,
    lead_time INTEGER, arrival_date_year INTEGER, arrival_date_week_number INTEGER,
    arrival_date_day_of_month INTEGER, stays_in_weekend_nights INTEGER, stays_in_week_nights INTEGER,
    adults INTEGER, children DOUBLE PRECISION, babies INTEGER, is_repeated_guest INTEGER,
    previous_cancellations INTEGER, previous_bookings_not_canceled INTEGER, agent DOUBLE PRECISION,
    company DOUBLE PRECISION, days_in_waiting_list INTEGER, adr DOUBLE PRECISION,
    total_of_special_requests INTEGER, rooms_in_booking INTEGER
)
WHERE p.replay_month = :month
ORDER BY p.id;

-- 2. Drift on predictions: distribution of probabilities per month and model version.
SELECT replay_month, model_version, count(*) AS rows,
       avg(probability) AS mean_probability,
       avg(is_canceled::int) AS predicted_cancel_rate
FROM prediction_logs
GROUP BY replay_month, model_version
ORDER BY replay_month, model_version;

-- 3. Performance once labels are released: metrics inputs per month and model version.
SELECT p.replay_month, p.model_version, count(*) AS labeled_rows,
       avg(l.is_canceled) AS true_cancel_rate,
       avg(((p.is_canceled)::int = l.is_canceled)::int) AS accuracy
FROM prediction_logs AS p
JOIN labels AS l USING (booking_id)
GROUP BY p.replay_month, p.model_version
ORDER BY p.replay_month, p.model_version;

-- 4. Retraining data: the latest logged input of every labeled booking, with its label.
SELECT DISTINCT ON (p.booking_id) p.booking_id, l.month, p.input, l.is_canceled
FROM prediction_logs AS p
JOIN labels AS l USING (booking_id)
WHERE l.month <= :month
ORDER BY p.booking_id, p.created_at DESC;

-- 5. Input warnings per month: unseen categories and values outside the train range.
SELECT p.replay_month, w ->> 'column' AS column_name, w ->> 'issue' AS issue, count(*) AS requests
FROM prediction_logs AS p
CROSS JOIN LATERAL jsonb_array_elements(p.warnings) AS w
GROUP BY 1, 2, 3
ORDER BY 1, 4 DESC;
