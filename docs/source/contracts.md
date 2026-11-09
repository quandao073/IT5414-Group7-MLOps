# Hợp đồng giữa các module

> **Phiên bản 0.1** (09/11/2026) · Người liên hệ: Bùi Trung Đức (Serving)
>
> Tài liệu này ghi các quy ước mà nhiều module cùng dựa vào: request/response của API, định dạng artifact mô hình,
> dữ liệu replay, biến môi trường và Docker. Ví dụ chạy được nằm trong `docs/examples/`.

## Quy tắc thay đổi

- Muốn đổi một mục: mở PR sửa file này và tag người liên hệ của mục đó, hoặc nhắn trực tiếp.
- Khi một mục đã có người dùng, chỉ **thêm** (trường không bắt buộc, cột mới). Không đổi tên, không xóa, không đổi
  kiểu. Thay đổi phá vỡ tương thích phải tăng phiên bản và báo cả nhóm.
- Giá trị dùng chung (đường dẫn, tên model, URL) đọc từ biến môi trường (C5), không hard-code.

| Mã | Mục | Trạng thái | Người làm theo |
|---|---|---|---|
| C1 | [API request/response](#c1-api) | Đề xuất, đã có code | Frontend, load test |
| C2 | [Artifact mô hình](#c2-artifact-mo-hinh) | Đề xuất, đã có code | Training |
| C3 | [`rooms_in_booking` khi suy luận](#c3-rooms_in_booking) | Đề xuất, chờ Training xác nhận | Training, Frontend |
| C4 | Dữ liệu replay và nhãn trễ | Sẽ bổ sung | Data, Drift |
| C5 | [Biến môi trường](#c5-bien-moi-truong) | Đề xuất | Platform |
| C6 – C11 | MLflow, `/reload`, ONNX, bảng log, metrics, compose | Sẽ bổ sung | – |
| C12 | [Docker image](#c12-docker) | Đề xuất, đã có Dockerfile | Platform, Training |
| C13 | Checklist demo | Sẽ bổ sung | Cả nhóm |

## C1. API

Base URL khi chạy local: `http://localhost:8000`. Swagger UI: `/docs`. Schema pydantic: `src/hotel_booking/schemas.py`.

| Method | Đường dẫn | Body | Response |
|---|---|---|---|
| `GET` | `/health` | – | `{"status": "ok", "model_loaded": bool, "model_version": str \| null}` |
| `POST` | `/predict` | Một `Booking` | `{"probability", "is_canceled", "threshold", "model_version", "warnings"}` |
| `POST` | `/predict/batch` | `{"bookings": [Booking, ...]}` (1 – 10 000 bản ghi) | `{"predictions": [{"probability", "is_canceled"}], "threshold", "model_version", "warnings"}` |

**`Booking`** gồm đúng các cột `config.MODEL_INPUT_COLUMNS`: 25 cột thô biết được lúc đặt phòng và
`rooms_in_booking`. Không gửi các cột leakage (`reservation_status`, `reservation_status_date`) hay các cột chỉ biết
sau khi đặt (`assigned_room_type`, `booking_changes`, `required_car_parking_spaces`, `country`); trường lạ bị từ chối.

| Nhóm | Cột | Kiểu |
|---|---|---|
| Phân loại | `hotel`, `arrival_date_month`, `meal`, `market_segment`, `distribution_channel`, `reserved_room_type`, `deposit_type`, `customer_type` | chuỗi |
| Số nguyên | `lead_time`, `arrival_date_year`, `arrival_date_week_number`, `arrival_date_day_of_month`, `stays_in_weekend_nights`, `stays_in_week_nights`, `adults`, `babies`, `is_repeated_guest`, `previous_cancellations`, `previous_bookings_not_canceled`, `days_in_waiting_list`, `total_of_special_requests` | số nguyên ≥ 0 (tuần 1–53, ngày 1–31, `is_repeated_guest` 0/1) |
| Số thực, có thể `null` | `children`, `agent`, `company` | số hoặc `null` |
| Số thực | `adr` | số |
| Không bắt buộc | `rooms_in_booking` | số nguyên ≥ 1, xem C3 |

**Xác suất** là `predict_proba[:, 1]`, xác suất bị hủy. `is_canceled = probability >= threshold`, với `threshold` lấy
từ model card (C2).

**Lỗi:**

- `422`: request sai định dạng (sai kiểu, thiếu cột, giá trị không thể có, trường lạ). Body là `detail` chuẩn của
  FastAPI, `loc` chỉ tên cột lỗi. Ví dụ: `docs/examples/predict_422_response.json`.
- `503`: API chưa nạp được mô hình.
- Giá trị đúng định dạng nhưng **chưa gặp trong train** (category mới, số ngoài khoảng train) vẫn được dự đoán và
  báo trong `warnings` (`column`, `issue`, `rows`), vì đây chính là thứ giám sát drift cần thấy. `issue` lấy từ
  `data.validate_inputs`: `unseen category`, `outside train range`.

Ví dụ: `predict_request.json`, `predict_response.json`, `predict_batch_request.json`, `predict_batch_response.json`.

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @docs/examples/predict_request.json
```

## C2. Artifact mô hình

Một mô hình được phục vụ gồm hai file trong cùng thư mục (mặc định `models/`, đã được `.gitignore` bỏ qua, không
commit):

| File | Nội dung |
|---|---|
| `model.joblib` | Object đã fit, dump nguyên bằng `joblib.dump` (ví dụ kết quả `model.build_from_spec(spec).fit(...)`). Có `predict_proba(X)` với `X` là DataFrame **thô** gồm các cột `MODEL_INPUT_COLUMNS` |
| `model_card.json` | Cùng định dạng `reports/it3190_model_card.json`, thêm trường `model_version` |

Trường của model card mà API đọc:

| Trường | Kiểu | Dùng để |
|---|---|---|
| `model_version` | chuỗi | Trả về trong mọi response, ghi log |
| `decision_threshold` | số | Quyết định `is_canceled` |
| `input_schema` | object, kết quả `data.build_input_schema(train)` | Sinh `warnings` |

Các trường khác (`model_name`, `created_at`, `model_spec`, `test_metrics`, `data`, `environment`...) giữ như model
card IT3190 để tra cứu.

Tạo mô hình mẫu theo đúng định dạng:

```bash
uv run invoke preprocess-data
uv run invoke sample-model                                     # Logistic Regression
uv run invoke sample-model --spec configs/model/champion.yaml  # ensemble champion
```

`src/hotel_booking/sample_model.py` chọn ngưỡng tốt nhất theo F1 trên 3 tháng cuối của train (không dùng test), rồi
fit lại trên toàn bộ train. Khi Training lưu mô hình thật, chỉ cần giữ hai file trên đúng định dạng. Cách đăng ký lên
MLflow Registry sẽ được bổ sung ở C6.

## C3. `rooms_in_booking`

`rooms_in_booking` là số phòng trong cùng một lượt đặt; lúc train nó được đếm bằng `data.add_group_size` (số dòng
trùng mọi cột đầu vào).

- Client nên gửi giá trị này nếu biết (như fixture `booking` trong `tests/conftest.py`).
- Nếu bỏ trống: API đếm các bản ghi giống hệt nhau **trong cùng request**. Một bản ghi lẻ qua `/predict` nhận giá trị 1.
  Một nhóm phòng gửi chung qua `/predict/batch` được đếm đúng như lúc train.

## C5. Biến môi trường

Danh sách đầy đủ trong `.env.example` ở gốc repo. Bí mật (mật khẩu, token) chỉ đặt trong `.env`, không commit.

| Biến | Mặc định | Dùng bởi |
|---|---|---|
| `MODEL_PATH` | `models/model.joblib` (trong container: `/app/models/model.joblib`) | API |
| `MODEL_CARD_PATH` | `models/model_card.json` (trong container: `/app/models/model_card.json`) | API |
| `API_URL` | `http://localhost:8000` | Replay, Frontend, load test |

## C12. Docker

| Image | Dockerfile | Lệnh chạy |
|---|---|---|
| `api:latest` | `dockerfiles/api.dockerfile` | `docker run --rm -p 8000:8000 -v ./models:/app/models:ro api:latest` |
| `train:latest` | `dockerfiles/train.dockerfile` | `docker run --rm -v ./data:/app/data -v ./models:/app/models train:latest` |

- Build cả hai: `uv run invoke docker-build`.
- Image dựa trên `python:3.12-slim-bookworm` (có `libgomp1` cho LightGBM/XGBoost), multi-stage với `uv sync --frozen
  --no-dev`, nên chỉ cài dependency chính trong `uv.lock`.
- Mô hình **không** đóng gói trong image API mà được mount vào `/app/models` (sau này nạp từ MLflow Registry).
- Image train chạy `python -m hotel_booking.train` và có sẵn `configs/`; Training giữ cho `train.py` chạy được bằng
  lệnh này, dữ liệu mount vào `/app/data`.
