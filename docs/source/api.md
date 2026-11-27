# API

API dự báo xác suất hủy đặt phòng, viết bằng FastAPI. Khi chạy local, base URL là
`http://localhost:8000`; Swagger UI ở [`/docs`](http://localhost:8000/docs) và OpenAPI schema
ở `/openapi.json`.

Hợp đồng chi tiết và quy tắc thay đổi nằm ở [Hợp đồng](contracts.md#c1-api); trang này là
bản tóm tắt để dùng hằng ngày.

## Endpoint

| Method | Đường dẫn | Body | Response |
|---|---|---|---|
| `GET` | `/health` | – | `status`, `model_loaded`, `model_version`, `prediction_logging` |
| `POST` | `/predict` | Một `Booking` | `probability`, `is_canceled`, `threshold`, `model_version`, `warnings` |
| `POST` | `/predict/batch` | `{"bookings": [Booking, ...]}` (1–10 000) | `predictions`, `threshold`, `model_version`, `warnings` |
| `POST` | `/reload` | – | `previous_version`, `current_version`, `changed`, `reloaded_at` |
| `GET` | `/metrics` | – | Prometheus text format |

`probability` là xác suất bị hủy (`predict_proba[:, 1]`).
`is_canceled = probability >= threshold`, với `threshold` đọc từ model card.

## Schema `Booking`

`Booking` nhận đúng 25 cột thô biết được lúc đặt phòng, cộng `rooms_in_booking`. Trường lạ bị
từ chối với mã `422`.

| Nhóm | Cột | Kiểu |
|---|---|---|
| Phân loại | `hotel`, `arrival_date_month`, `meal`, `market_segment`, `distribution_channel`, `reserved_room_type`, `deposit_type`, `customer_type` | chuỗi |
| Số nguyên | `lead_time`, `arrival_date_year`, `arrival_date_week_number`, `arrival_date_day_of_month`, `stays_in_weekend_nights`, `stays_in_week_nights`, `adults`, `babies`, `is_repeated_guest`, `previous_cancellations`, `previous_bookings_not_canceled`, `days_in_waiting_list`, `total_of_special_requests` | số nguyên ≥ 0 (tuần 1–53, ngày 1–31, `is_repeated_guest` 0/1) |
| Số thực, có thể `null` | `children`, `agent`, `company` | số hoặc `null` |
| Số thực | `adr` | số |
| Không bắt buộc | `rooms_in_booking` | số nguyên ≥ 1 |

Chỉ để ghi log, không đưa vào mô hình:

- `booking_id` (không bắt buộc) trong mỗi `Booking`, để nối với nhãn về sau.
- Header `X-Replay-Month: YYYY-MM` (không bắt buộc) cho `/predict` và `/predict/batch`.

`rooms_in_booking` là số phòng trong cùng một lượt đặt. Nếu bỏ trống, API đếm các bản ghi
giống hệt nhau trong cùng request; một bản ghi lẻ nhận giá trị `1`.

## Ví dụ

Request mẫu:
[`predict_request.json`](https://github.com/quandao073/IT5414-Group7-MLOps/blob/develop/docs/examples/predict_request.json)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @docs/examples/predict_request.json
```

Response:

```json
{
  "probability": 0.7312,
  "is_canceled": true,
  "threshold": 0.456,
  "model_version": "logreg-2026-11-27",
  "warnings": []
}
```

Batch:

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d @docs/examples/predict_batch_request.json
```

Các file mẫu khác:
[`predict_batch_request.json`](https://github.com/quandao073/IT5414-Group7-MLOps/blob/develop/docs/examples/predict_batch_request.json),
[`predict_response.json`](https://github.com/quandao073/IT5414-Group7-MLOps/blob/develop/docs/examples/predict_response.json),
[`predict_422_response.json`](https://github.com/quandao073/IT5414-Group7-MLOps/blob/develop/docs/examples/predict_422_response.json),
[`stream_sample.csv`](https://github.com/quandao073/IT5414-Group7-MLOps/blob/develop/docs/examples/stream_sample.csv).

## Lỗi và cảnh báo

| Mã | Khi nào | Body |
|---|---|---|
| `422` | Sai kiểu, thiếu cột, giá trị không thể có, trường lạ | `detail` chuẩn của FastAPI, `loc` chỉ tên cột lỗi |
| `503` | API chưa nạp được mô hình, hoặc `/reload` thất bại | `detail` ghi lý do |

Giá trị đúng định dạng nhưng **chưa gặp trong train** (category mới, số ngoài khoảng train)
vẫn được dự đoán và báo trong `warnings`:

```json
{
  "warnings": [
    {"column": "country", "issue": "unseen category", "rows": 1}
  ]
}
```

`issue` nhận một trong `unseen category` hoặc `outside train range`. Đây là tín hiệu drift,
không phải lỗi request.

## Nạp lại mô hình

`POST /reload` nạp lại mô hình từ nguồn hiện tại mà không restart container:

```bash
curl -X POST http://localhost:8000/reload -H "X-Reload-Token: $RELOAD_TOKEN"
```

| Mã | Ý nghĩa |
|---|---|
| `200` | `previous_version`, `current_version`, `changed`, `reloaded_at` |
| `401` | Thiếu hoặc sai `X-Reload-Token` (chỉ khi `RELOAD_TOKEN` được đặt) |
| `503` | Không nạp được; **mô hình cũ vẫn được phục vụ** |

Request đang xử lý dùng trọn mô hình lúc bắt đầu; request sau khi `/reload` trả về dùng mô
hình mới. `changed: false` nghĩa là nguồn vẫn chứa đúng phiên bản đang chạy.

## Metrics

`GET /metrics` trả định dạng Prometheus. Các metric chính:

| Metric | Nhãn | Ý nghĩa |
|---|---|---|
| `api_requests_total` | `endpoint`, `status` | Số request theo route và mã trạng thái |
| `api_request_latency_seconds` | `endpoint` | Độ trễ trả lời |
| `api_errors_total` | `endpoint`, `type` | Request lỗi |
| `api_prediction_probability` | – | Phân bố xác suất dự đoán |
| `api_predictions_total` | `outcome` | Số booking theo quyết định |
| `api_input_warnings_total` | `column`, `issue` | Giá trị chưa gặp trong train |
| `api_model_info` | `model_version`, `backend` | Phiên bản mô hình đang phục vụ |
| `api_model_reloads_total` | `result` | Kết quả `/reload` |
| `api_prediction_log_errors_total` | – | Lỗi ghi log dự đoán |

Chi tiết xem [Hợp đồng](contracts.md#c10-metrics-prometheus).
