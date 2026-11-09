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
| C4 | [Dữ liệu replay và nhãn trễ](#c4-du-lieu-replay) | Đề xuất, đã có code | Data, Drift |
| C5 | [Biến môi trường](#c5-bien-moi-truong) | Đề xuất | Platform |
| C6 | MLflow Registry | Sẽ bổ sung | Training |
| C7 | Endpoint `/reload` | Sẽ bổ sung | Training (DAG retrain) |
| C8 | Backend ONNX | Sẽ bổ sung | Load test |
| C9 | [Bảng log dự đoán và nhãn](#c9-bang-log-du-doan-va-nhan) | Đề xuất, SQL đã chạy thử, API chưa ghi | Drift, Training, Platform |
| C10 | Metrics Prometheus | Sẽ bổ sung | Platform |
| C11 | Service API trong compose | Sẽ bổ sung | Platform |
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

## C4. Dữ liệu replay

Replay mô phỏng production bằng các đặt phòng có ngày đến năm 2017: `test` (01–04/2017) và `production_holdout`
(05–08/2017).

| Mục | Quy ước |
|---|---|
| Batch theo tháng | `data/stream/YYYY-MM.csv` (`2017-01` … `2017-08`), cùng cột với `data/processed/test.csv`, có `booking_id`, `arrival_date`, `is_canceled`. Thư mục này đã được `.gitignore` bỏ qua |
| Tạo batch | `replay.py` tự tạo từ `data/processed/` nếu thư mục chưa có batch; file đã có thì giữ nguyên, nên Data có thể tự sinh batch thay thế miễn đúng tên file và cột |
| Gửi | Mỗi tháng gửi lần lượt vào `POST /predict/batch`, cấu hình `--batch-size` (mặc định 200) và `--delay` (giây giữa hai request) |
| Nhãn trễ | Nhãn của tháng M chỉ được ghi **sau khi** đã gửi xong tháng M, vào `data/stream/labels/YYYY-MM.csv` với cột `booking_id`, `is_canceled`, `month`, `released_at`. Khi có PostgreSQL, nhãn chuyển sang bảng `labels` (C9) |

Ví dụ batch: `docs/examples/stream_sample.csv` (20 dòng dữ liệu **giả**, chỉ để xem cột).

```bash
uv run invoke serve-api            # cửa sổ 1
uv run invoke replay --delay 0.5   # cửa sổ 2, hoặc --months 2017-01,2017-02
```

## C5. Biến môi trường

Danh sách đầy đủ trong `.env.example` ở gốc repo. Bí mật (mật khẩu, token) chỉ đặt trong `.env`, không commit.

| Biến | Mặc định | Dùng bởi |
|---|---|---|
| `MODEL_PATH` | `models/model.joblib` (trong container: `/app/models/model.joblib`) | API |
| `MODEL_CARD_PATH` | `models/model_card.json` (trong container: `/app/models/model_card.json`) | API |
| `API_URL` | `http://localhost:8000` | Replay, Frontend, load test |

## C9. Bảng log dự đoán và nhãn

PostgreSQL. DDL: `docs/examples/sql/schema.sql`, truy vấn mẫu: `docs/examples/sql/queries.sql` (đã chạy thử trên
PostgreSQL 16). API sẽ chạy DDL lúc khởi động (`CREATE TABLE IF NOT EXISTS`), nên không cần migration riêng; ai cũng
có thể chạy lại file này mà không mất dữ liệu.

**`prediction_logs`**: mỗi bản ghi được chấm là một dòng; một request batch ghi nhiều dòng cùng `request_id`.

| Cột | Kiểu | Ý nghĩa |
|---|---|---|
| `id` | `BIGSERIAL` PK | |
| `request_id` | `UUID` | Một request HTTP |
| `row_index` | `INTEGER` | Vị trí bản ghi trong request (0 với `/predict`) |
| `created_at` | `TIMESTAMPTZ` | Lúc ghi log |
| `endpoint` | `TEXT` | `/predict` hoặc `/predict/batch` |
| `model_version` | `TEXT` | Như trong response (C1, C2) |
| `backend` | `TEXT` | `joblib` hoặc `onnx` (C8) |
| `booking_id` | `BIGINT`, có thể `NULL` | Khóa nối với `labels`; replay luôn gửi, client thường có thể bỏ trống |
| `replay_month` | `TEXT`, có thể `NULL` | Tháng `YYYY-MM` khi request đến từ replay |
| `input` | `JSONB` | Bản ghi đầu vào đúng như mô hình nhận: các cột `MODEL_INPUT_COLUMNS`, đã điền `rooms_in_booking` |
| `probability` | `DOUBLE PRECISION` | Xác suất hủy, trong [0, 1] |
| `threshold` | `DOUBLE PRECISION` | Ngưỡng lúc dự đoán |
| `is_canceled` | `BOOLEAN` | Quyết định của mô hình |
| `latency_ms` | `DOUBLE PRECISION` | Thời gian xử lý request (cùng giá trị cho mọi dòng của một batch) |
| `warnings` | `JSONB` | Mảng `warnings` của response (C1) |

**`labels`**: nhãn thật, replay ghi sau khi gửi xong cả tháng (C4).

| Cột | Kiểu | Ý nghĩa |
|---|---|---|
| `booking_id` | `BIGINT` PK | Như `booking_id` của `data/processed/*.csv` |
| `is_canceled` | `SMALLINT` (0/1) | Nhãn thật |
| `month` | `TEXT` | Tháng `YYYY-MM` được mở nhãn |
| `released_at` | `TIMESTAMPTZ` | Lúc mở nhãn |

**Truy vấn mẫu** (`queries.sql`):

1. Đầu vào của một tháng, tách thành cột: cho drift service so với dữ liệu tham chiếu `data/processed/train.csv`.
2. Phân bố xác suất theo tháng và phiên bản mô hình.
3. Độ chính xác theo tháng sau khi đã có nhãn.
4. Dữ liệu có nhãn để retrain: bản ghi mới nhất của mỗi `booking_id` đã có nhãn.
5. Số request có cảnh báo đầu vào, theo tháng và cột.

**Thay đổi đi kèm ở C1** (sẽ làm cùng việc ghi log, chỉ thêm, không phá request hiện tại): mỗi `Booking` nhận thêm
trường không bắt buộc `booking_id`, và request có thể kèm header `X-Replay-Month: YYYY-MM`. Hai giá trị này chỉ ghi
vào log, không đưa vào mô hình.

Kết nối: biến môi trường `DATABASE_URL` (ví dụ `postgresql+psycopg://user:password@postgres:5432/hotel`). Không
có `DATABASE_URL` thì API vẫn chạy, chỉ không ghi log.

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
