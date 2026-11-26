# PromQL cho dashboard API

Truy vấn mẫu cho các panel Grafana dựa trên metrics của API (hợp đồng C10 trong `docs/source/contracts.md`).
`$__rate_interval` là biến có sẵn của Grafana.

| Panel | Loại | PromQL |
|---|---|---|
| Request mỗi giây theo route | Time series | `sum by (endpoint) (rate(api_requests_total{endpoint!="/metrics"}[$__rate_interval]))` |
| Tỉ lệ lỗi | Time series / Stat | `sum(rate(api_errors_total[$__rate_interval])) / sum(rate(api_requests_total{endpoint!="/metrics"}[$__rate_interval]))` |
| Lỗi theo loại | Time series | `sum by (type) (rate(api_errors_total[$__rate_interval]))` |
| Latency p50 / p95 / p99 | Time series | `histogram_quantile(0.95, sum by (le) (rate(api_request_latency_seconds_bucket{endpoint=~"/predict.*"}[$__rate_interval])))` (đổi 0.95 thành 0.5, 0.99) |
| Phân bố xác suất | Heatmap / Bar gauge | `sum by (le) (increase(api_prediction_probability_bucket[$__range]))` |
| Xác suất trung bình | Time series | `rate(api_prediction_probability_sum[$__rate_interval]) / rate(api_prediction_probability_count[$__rate_interval])` |
| Tỉ lệ dự đoán hủy | Time series | `sum(rate(api_predictions_total{outcome="canceled"}[$__rate_interval])) / sum(rate(api_predictions_total[$__rate_interval]))` |
| Cảnh báo đầu vào theo cột | Table / Bar chart | `sum by (column, issue) (increase(api_input_warnings_total[$__range]))` |
| Mô hình đang phục vụ | Stat (hiển thị label) | `api_model_info == 1` |
| API còn sống | Stat | `up{job="api"}` |

**Cảnh báo gợi ý:**

| Cảnh báo | Điều kiện | Trong |
|---|---|---|
| API không trả lời | `up{job="api"} == 0` | 1 phút |
| Tỉ lệ lỗi cao | Tỉ lệ lỗi > 0,05 | 5 phút |
| Latency cao | p95 của `/predict.*` > 0,5 giây | 5 phút |
| Không có mô hình | `absent(api_model_info == 1)` | 1 phút |
| Xác suất trung bình lệch | Xác suất trung bình ngoài khoảng 0,25 – 0,55 | 30 phút |
