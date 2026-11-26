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
| C6 | [MLflow Registry](#c6-mlflow-registry) | Đề xuất, đã có code (chưa nối vào API) | Training |
| C7 | [Endpoint `/reload`](#c7-endpoint-reload) | Đề xuất, đã có code | Training (DAG retrain) |
| C8 | [Backend ONNX](#c8-backend-onnx) | Đề xuất, đã có code | Load test, Platform |
| C9 | [Bảng log dự đoán và nhãn](#c9-bang-log-du-doan-va-nhan) | Đề xuất, đã có code, chạy thử trên PostgreSQL 16 | Drift, Training, Platform |
| C10 | [Metrics Prometheus](#c10-metrics-prometheus) | Đề xuất, đã có code | Platform |
| C11 | [Service trong docker compose](#c11-docker-compose) | Đề xuất, đã có `docker-compose.yml` | Platform, cả nhóm |
| C12 | [Docker image](#c12-docker) | Đề xuất, đã có Dockerfile | Platform, Training |
| C13 | Checklist demo | Sẽ bổ sung | Cả nhóm |
| C14 | [Drift service và DAG `monitor_drift`](#c14-drift-service) | Đề xuất, đã có code; API ghi log từ C9 | Platform (compose, Grafana), Training (DAG `retrain`) |

## C1. API

Base URL khi chạy local: `http://localhost:8000`. Swagger UI: `/docs`. Schema pydantic: `src/hotel_booking/schemas.py`.

| Method | Đường dẫn | Body | Response |
|---|---|---|---|
| `GET` | `/health` | – | `{"status": "ok", "model_loaded": bool, "model_version": str \| null, "prediction_logging": bool}` |
| `POST` | `/predict` | Một `Booking` | `{"probability", "is_canceled", "threshold", "model_version", "warnings"}` |
| `POST` | `/predict/batch` | `{"bookings": [Booking, ...]}` (1 – 10 000 bản ghi) | `{"predictions": [{"probability", "is_canceled"}], "threshold", "model_version", "warnings"}` |

**`Booking`** gồm đúng các cột `config.MODEL_INPUT_COLUMNS`: 25 cột thô biết được lúc đặt phòng và
`rooms_in_booking`. Không gửi các cột leakage (`reservation_status`, `reservation_status_date`) hay các cột chỉ biết
sau khi đặt (`assigned_room_type`, `booking_changes`, `required_car_parking_spaces`, `country`); trường lạ bị từ chối.

Ngoài ra, chỉ để ghi log (C9), không đưa vào mô hình:

- trường không bắt buộc `booking_id` (số nguyên) trong mỗi `Booking`, để nối với nhãn về sau;
- header không bắt buộc `X-Replay-Month: YYYY-MM` của `/predict` và `/predict/batch`, replay luôn gửi (sai định dạng
  trả `422`).

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
| `artifact_sha256` | chuỗi hex, không bắt buộc | Khi có, API chỉ nạp `model.joblib` có SHA-256 trùng giá trị này |

Khi thay mô hình đang chạy, ghi file mới dưới tên tạm rồi đổi tên (như `sample_model.save_artifacts`): ghi
`model.joblib.tmp` và `model_card.json.tmp` (card chứa `artifact_sha256` của file model mới), đổi tên model trước,
card sau. Nếu `/reload` chạy đúng lúc giữa hai lần đổi tên, checksum không khớp nên API giữ mô hình cũ; gọi lại là xong.

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
MLflow Registry xem C6; `train.py` ghi cùng định dạng này và `model_card.json` còn được log vào run MLflow.

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
| Nhãn trễ | Nhãn của tháng M chỉ được ghi **sau khi** đã gửi xong tháng M, vào `data/stream/labels/YYYY-MM.csv` với cột `booking_id`, `is_canceled`, `month`, `released_at`; khi có `DATABASE_URL` (hoặc `--database-url`) thì ghi thêm vào bảng `labels` (C9), chạy lại thì cập nhật chứ không nhân đôi |
| Định danh | Mỗi bản ghi gửi kèm `booking_id`, mỗi request kèm header `X-Replay-Month` (C1), nên log nối được với nhãn |

Ví dụ batch: `docs/examples/stream_sample.csv` (20 dòng dữ liệu **giả**, chỉ để xem cột).

```bash
uv run invoke serve-api            # cửa sổ 1
uv run invoke replay --delay 0.5   # cửa sổ 2, hoặc --months 2017-01,2017-02
```

## C5. Biến môi trường

Danh sách đầy đủ trong `.env.example` ở gốc repo. Bí mật (mật khẩu, token) chỉ đặt trong `.env`, không commit.

| Biến | Mặc định | Dùng bởi |
|---|---|---|
| `MODEL_SOURCE` | `file` (hiện chỉ có giá trị này; `mlflow` sẽ thêm ở C6) | API |
| `MODEL_BACKEND` | `joblib`; `onnx` để chạy bằng ONNX Runtime (C8) | API |
| `MODEL_PATH` | `models/model.joblib` (trong container: `/app/models/model.joblib`) | API |
| `MODEL_CARD_PATH` | `models/model_card.json` (trong container: `/app/models/model_card.json`) | API |
| `RELOAD_TOKEN` | trống: `/reload` không cần token | API, DAG retrain (bí mật, đặt trong `.env`) |
| `API_URL` | `http://localhost:8000` | Replay, Frontend, load test |
| `MLFLOW_TRACKING_URI` | `https://dagshub.com/quandao073/IT5414-Group7-MLOps.mlflow`; MLflow tự host: `http://localhost:5000` (trong compose: `http://mlflow:5000`) | Training, API (Registry) |
| `MLFLOW_TRACKING_USERNAME` | Tên tài khoản DagsHub của mỗi người | Training, API (Registry) |
| `CONFIG_DIR` | Thư mục `configs/` của Hydra; mặc định `configs/` cạnh `src/` (trong container: `/app/configs`) | Training (`train.py`, `tune.py`) |
| `MLFLOW_TRACKING_PASSWORD` | Access token DagsHub (bí mật, chỉ đặt trong `.env`) | Training, API (Registry) |
| `POSTGRES_USER`, `POSTGRES_PASSWORD` | `mlops`, `mlops` (chỉ dùng local) | docker compose: PostgreSQL, MLflow, Airflow |
| `S3_ACCESS_KEY`, `S3_SECRET_KEY` | `mlops`, `mlops-secret` (chỉ dùng local) | docker compose: SeaweedFS, MLflow; DVC remote `local` |
| `AIRFLOW_JWT_SECRET` | `local-only-jwt-secret` (chỉ dùng local) | docker compose: các service Airflow |
| `DATABASE_URL` | Không có (không ghi/đọc log). Trong compose: `postgresql+psycopg://mlops:mlops@postgres:5432/hotel` | API (ghi log, C9), replay (ghi nhãn, C4), Drift service (đọc log, C14) |
| `REFERENCE_DIR` | `data/processed` (trong container: `/app/data/processed`) | Drift service |
| `DRIFT_MIN_ROWS` | `100` | Drift service |
| `DRIFT_SERVICE_URL` | `http://drift:8001` | DAG `monitor_drift` |

## C7. Endpoint `/reload`

Nạp lại mô hình từ nguồn hiện tại (`MODEL_SOURCE`, C5) mà không khởi động lại container. Dùng ở cuối DAG retrain, sau
khi mô hình mới đã được ghi ra (C2) hoặc được gán alias `champion` (C6).

| Mục | Quy ước |
|---|---|
| Request | `POST /reload`, không có body. Nếu API đặt `RELOAD_TOKEN` thì gửi header `X-Reload-Token: <token>` |
| `200` | `{"previous_version": str \| null, "current_version": str, "changed": bool, "reloaded_at": "<ISO 8601 UTC>"}` |
| `401` | Thiếu hoặc sai `X-Reload-Token` |
| `503` | Không nạp được (thiếu file, sai checksum, card hỏng): **mô hình cũ vẫn được phục vụ**, `detail` ghi lý do |

- Request đang xử lý dùng trọn mô hình lúc bắt đầu; request đến sau khi `/reload` trả về dùng mô hình mới.
- Hai lần `/reload` cùng lúc chạy lần lượt.
- `changed: false` nghĩa là nguồn vẫn chứa đúng phiên bản đang chạy (ví dụ challenger không được promote).
- Kết quả được đếm trong `api_model_reloads_total{result}` (C10) và `api_model_info` đổi sang phiên bản mới.

Cách DAG kiểm tra: gọi `/reload`, kiểm tra mã `200` và `current_version` bằng phiên bản vừa promote; sau đó
`GET /health` cũng phải trả `model_version` đó. Ví dụ: `docs/examples/reload.http`.

```bash
curl -X POST http://localhost:8000/reload -H "X-Reload-Token: $RELOAD_TOKEN"
```

## C8. Backend ONNX

API chạy cùng một mô hình bằng hai backend, chọn bằng `MODEL_BACKEND` (C5): `joblib` (mặc định, scikit-learn) hoặc
`onnx` (ONNX Runtime). Request, response và `/reload` giống hệt nhau; chỉ `backend` trong `api_model_info` (C10) và
cột `backend` của log (C9) khác.

**Cách export:** `BookingFeatureBuilder` là code pandas nên vẫn chạy bằng Python; phần sau nó (impute, log1p, scale,
one-hot, bộ phân loại) thành đồ thị ONNX. Ensemble export mỗi thành viên thành một đồ thị và lấy trung bình xác suất
như `AveragingClassifier`. Mã nguồn: `src/hotel_booking/onnx_export.py`.

| Mục | Quy ước |
|---|---|
| Lệnh | `uv run invoke export-onnx` sau `invoke sample-model` (hoặc sau khi Training ghi mô hình mới, C2) |
| File | `models/model.onnx` cho mô hình đơn; `model.0.onnx`, `model.1.onnx`... theo thứ tự thành viên cho ensemble |
| Model card | Thêm mục `onnx`: `artifacts` (`file`, `sha256`), `opset` (17), `parity_rows`, `parity_max_abs_difference` |
| Kiểm tra | Lúc export so xác suất ONNX với joblib trên tối đa 5000 dòng của `test.csv`; test yêu cầu sai lệch < 1e-5 |
| Nạp | Backend `onnx` vẫn đọc `model.joblib` (lấy bộ tạo đặc trưng) và từ chối nạp nếu card thiếu mục `onnx` hoặc đồ thị không khớp `sha256` |
| Thay mô hình | Ghi mô hình mới (C2) rồi `invoke export-onnx` rồi `/reload` (C7). Card mới chưa có mục `onnx` thì API backend `onnx` giữ mô hình cũ |

Các điểm đã xử lý để export được mô hình của repo: converter riêng cho bước `log1p_nonnegative`, cho `OneHotEncoder`
có category hiếm (`min_frequency`, `handle_unknown="infrequent_if_exist"`), cho LightGBM và XGBoost (qua
`onnxmltools`). Cả `logistic_regression.yaml` và ensemble `champion.yaml` đều export được.

**So sánh:** `uv run invoke benchmark` đo trên `test.csv` thời gian gọi mô hình (tạo đặc trưng + suy luận, không gồm
HTTP) với 1 bản ghi và lô 1000 bản ghi (p50, p95, bản ghi/giây) và kích thước artifact, ghi `models/benchmark.json`
và in bảng Markdown để đưa vào báo cáo. Load test qua HTTP dùng hai cấu hình:

```bash
MODEL_BACKEND=joblib uv run invoke serve-api
MODEL_BACKEND=onnx uv run invoke serve-api
```

## C9. Bảng log dự đoán và nhãn

PostgreSQL. DDL: `docs/examples/sql/schema.sql`, truy vấn mẫu: `docs/examples/sql/queries.sql` (đã chạy thử trên
PostgreSQL 16). Khi có `DATABASE_URL`, API tạo các bảng còn thiếu lúc khởi động (bảng SQLAlchemy trong
`src/hotel_booking/prediction_log.py`, cùng cột với `schema.sql`, test kiểm tra hai bên khớp), nên không cần migration
riêng; `schema.sql` chạy lại cũng không mất dữ liệu.

Ghi log chạy trong background task sau khi đã trả response, nên không làm chậm request. Ghi lỗi (DB mất kết nối,
bảng bị xóa) chỉ ghi cảnh báo và tăng `api_prediction_log_errors_total` (C10), request vẫn thành công. Không mở được
DB lúc khởi động thì API vẫn chạy, `/health` báo `prediction_logging: false`.

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

`booking_id` và `replay_month` lấy từ trường `booking_id` và header `X-Replay-Month` của request (C1); hai giá trị này
không đưa vào mô hình.

Kết nối: biến môi trường `DATABASE_URL` (ví dụ `postgresql+psycopg://user:password@postgres:5432/hotel`). Không
có `DATABASE_URL` thì API vẫn chạy, chỉ không ghi log.

## C10. Metrics Prometheus

API expose metrics ở `GET /metrics` (định dạng text của Prometheus, không hiện trong Swagger). Mã nguồn:
`src/hotel_booking/monitoring.py`.

| Metric | Kiểu | Label | Ý nghĩa |
|---|---|---|---|
| `api_requests_total` | counter | `endpoint`, `status` | Số request theo route và mã trả về |
| `api_request_latency_seconds` | histogram | `endpoint` | Thời gian trả lời, bucket 5 ms – 10 s |
| `api_errors_total` | counter | `endpoint`, `type` | Request lỗi: `validation` (422), `unavailable` (503), `client` (4xx khác), `internal` (5xx khác) |
| `api_prediction_probability` | histogram | – | Xác suất hủy của từng bản ghi được chấm, bucket 0,1 |
| `api_predictions_total` | counter | `outcome` | Số bản ghi theo quyết định: `canceled`, `not_canceled` |
| `api_input_warnings_total` | counter | `column`, `issue` | Số bản ghi có giá trị chưa gặp trong train (như `warnings` của C1) |
| `api_model_info` | gauge | `model_version`, `backend` | Bằng 1 cho mô hình đang phục vụ; không có dòng nào khi chưa nạp mô hình |
| `api_model_reloads_total` | counter | `result` | Số lần gọi `/reload`: `success`, `failure` (C7) |
| `api_prediction_log_errors_total` | counter | – | Số lần ghi log dự đoán thất bại (C9) |

- `endpoint` là mẫu đường dẫn của route (`/predict`, `/predict/batch`...); đường dẫn không tồn tại gộp thành
  `unmatched`, nên số giá trị label không tăng vô hạn.
- Một request batch tính 1 lần trong `api_requests_total` nhưng N lần trong `api_prediction_probability`.
- Metric nằm trong bộ nhớ của từng tiến trình API; khởi động lại thì counter về 0 (Prometheus xử lý bằng `rate`).

Mẫu cho Platform: đoạn scrape `docs/examples/prometheus_scrape.yml`, PromQL cho panel và cảnh báo
`docs/examples/grafana_queries.md`.

## C11. Docker compose

`docker compose up -d --build` ở gốc repo. Trong mạng compose, service gọi nhau bằng tên service (cột URL nội bộ).

| Service | Cổng host | URL nội bộ | Vai trò |
|---|---|---|---|
| `api` | 8000 | `http://api:8000` | API dự đoán, mount `./models` vào `/app/models` |
| `frontend` | 8501 | – | Streamlit, gọi `API_URL=http://api:8000` |
| `mlflow` | 5000 | `http://mlflow:5000` | Tracking + Model Registry; artifact đi qua server (`--serve-artifacts`) nên client không cần key S3 |
| `postgres` | 5432 | `postgres:5432` | Database `mlflow`, `airflow`, `hotel` (bảng C9, `DATABASE_URL=...@postgres:5432/hotel`) |
| `seaweedfs` | 8333 | `http://seaweedfs:8333` | S3: bucket `mlflow` (artifact), `dvc` (DVC remote `local`) |
| `storage-init` | – | – | Chạy một lần, tạo bucket |
| `airflow-apiserver` | 8080 | `http://airflow-apiserver:8080` | Airflow 3 UI + API (không đăng nhập, chỉ dùng local) |
| `airflow-scheduler`, `airflow-dag-processor` | – | – | LocalExecutor, đọc DAG từ `./airflow/dags` |
| `airflow-init` | – | – | Chạy một lần, `airflow db migrate` vào database `airflow` |

- Dữ liệu nằm trong volume `postgres-data`, `seaweedfs-data`; `docker compose down -v` xóa sạch.
- `infra/postgres/init.sql` chỉ chạy khi volume `postgres-data` được tạo lần đầu. Thêm database mới thì cần
  `docker compose down -v` hoặc tạo tay bằng `psql`.
- Service mới (Prometheus, Grafana, drift) thêm vào cùng file và dùng lại `postgres`/`seaweedfs`.
- DAG dùng image `apache/airflow:3.4.0` nguyên bản, không import `hotel_booking`: gọi service qua HTTP
  (ví dụ `/reload` của API) hoặc chạy image `train` bằng `DockerOperator`. CI (`Airflow DAGs`) import mọi DAG
  bằng `DagBag` trong đúng image này.

## C12. Docker

| Image | Dockerfile | Lệnh chạy |
|---|---|---|
| `api:latest` | `dockerfiles/api.dockerfile` | `docker run --rm -p 8000:8000 -v ./models:/app/models:ro api:latest` |
| `train:latest` | `dockerfiles/train.dockerfile` | `docker run --rm -v ./data:/app/data -v ./models:/app/models train:latest` |
| `hotel-booking-frontend` | `dockerfiles/frontend.dockerfile` | `docker compose up -d frontend` |
| `hotel-booking-mlflow` | `dockerfiles/mlflow.dockerfile` | `docker compose up -d mlflow` (image MLflow chính thức + `psycopg2`, `boto3`) |
| `drift:latest` | `dockerfiles/drift.dockerfile` | `docker run --rm -p 8001:8001 -v ./data/processed:/app/data/processed:ro -e DATABASE_URL=... drift:latest` |

- Build tất cả: `uv run invoke docker-build`.
- Image dựa trên `python:3.12-slim-bookworm` (có `libgomp1` cho LightGBM/XGBoost), multi-stage với `uv sync --frozen
  --no-dev`, nên chỉ cài dependency chính trong `uv.lock`.
- Mô hình **không** đóng gói trong image API mà được mount vào `/app/models` (sau này nạp từ MLflow Registry).
- Image train chạy `python -m hotel_booking.train` và có sẵn `configs/`; Training giữ cho `train.py` chạy được bằng
  lệnh này, dữ liệu mount vào `/app/data`.

## C14. Drift service

Service FastAPI riêng (`src/hotel_booking/drift_service.py`, cổng 8001), đọc bảng `prediction_logs` (C9) và so đầu
vào đã log với dữ liệu tham chiếu `train.csv` bằng Evidently, cùng cách tính với báo cáo offline
`reports/drift/` (`drift.column_drift`). Chạy local: `uv run invoke serve-drift`, Swagger UI tại `/docs`.

| Endpoint | Ý nghĩa |
|---|---|
| `GET /health` | `status`, `reference_rows`, `database` (có `DATABASE_URL` hay không) |
| `GET /drift?month=YYYY-MM` | Drift của một tháng replay (`replay_month`) |
| `GET /drift?start=...&end=...` | Drift theo `created_at` trong `[start, end)`, thời gian ISO 8601 |
| `GET /drift/latest` | Drift của tháng replay mới nhất; DAG `monitor_drift` gọi endpoint này |
| `GET /drift/report?...` | Report HTML của Evidently, cùng tham số với `/drift` |
| `GET /metrics` | Metrics Prometheus |

Response của `/drift`: `window`, `rows`, `reference_rows`, `columns`, `drifted_columns`, `drift_share` (tỉ lệ cột
bị drift, 0–1), `dataset_drift` (`drift_share >= 0.5`) và `details` (mỗi cột: `column`, `method`, `threshold`,
`score`, `drifted`).

- Không so các cột lịch (`arrival_date_year`, `_month`, `_week_number`, `_day_of_month`) vì chúng luôn khác khi so
  hai giai đoạn; `agent`, `company` được so như biến phân loại.
- Cửa sổ có ít hơn `DRIFT_MIN_ROWS` dòng trả `404`; không có `DATABASE_URL` thì các endpoint `/drift*` trả `503`;
  thiếu hoặc trộn `month` với `start`/`end` trả `422`.
- Metrics (label `window` là tháng replay, hoặc `custom` cho khoảng `created_at`): `drift_share`,
  `drift_drifted_columns`, `drift_rows`, `drift_dataset_drift`, `drift_last_check_timestamp_seconds` và
  `drift_checks_total`. Grafana cảnh báo theo `drift_share`.

**Trong docker compose** (Platform thêm khi gộp compose):

```yaml
  drift:
    build:
      context: .
      dockerfile: dockerfiles/drift.dockerfile
    ports: ["8001:8001"]
    environment:
      DATABASE_URL: postgresql+psycopg://${POSTGRES_USER:-mlops}:${POSTGRES_PASSWORD:-mlops}@postgres:5432/hotel
    volumes:
      - ./data/processed:/app/data/processed:ro
    depends_on:
      postgres: {condition: service_healthy}
    restart: unless-stopped
```

Prometheus scrape `drift:8001/metrics`.

**DAG `monitor_drift`** (`airflow/dags/monitor_drift.py`): chạy theo lịch `MONITOR_DRIFT_SCHEDULE` (mặc định mỗi 10
phút), gọi `GET /drift/latest`; nếu `drift_share` đạt Airflow Variable `drift_share_threshold` (mặc định 0.5) và
tháng đó chưa kích hoạt lần nào (Variable `monitor_drift_last_window`) thì trigger DAG có id **`retrain`** với
`conf = {"reason": "drift", "window": "YYYY-MM", "drift_share": "..."}`. Training đặt đúng id `retrain` cho DAG
huấn luyện lại; cho tới khi có DAG đó, bước `trigger_retrain` sẽ báo lỗi.

## C6. MLflow Registry

Code: `src/hotel_booking/train.py` (log và đăng ký), `src/hotel_booking/registry.py` (alias, tải model).

| Mục | Giá trị |
|---|---|
| Tên model đăng ký | `hotel-booking-cancellation` |
| Alias | `challenger`: bản vừa đăng ký; `champion`: bản đang phục vụ. Không dùng stage cũ |
| URI nạp | `models:/hotel-booking-cancellation@champion` (`registry.load_model_by_alias`) |
| Flavor | `mlflow.sklearn`, định dạng `cloudpickle`; nhận các cột `config.MODEL_INPUT_COLUMNS` |
| Artifact của run | `model/` (model), `model_card.json`, `config/config.yaml`, `plots/` |
| Metric của run | `roc_auc`, `pr_auc`, `f1`, `average_precision`, `decision_threshold`, ... trên split `test` |

Quy trình:

```bash
uv run invoke train --overrides "model=lgbm train.register=true"   # đăng ký bản mới là challenger
uv run invoke retrain                                              # train + đăng ký + promote theo configs/promotion/
uv run invoke registry --command "promote --metric roc_auc"        # promote tay nếu tốt hơn champion
uv run invoke registry --command "show"
```

Metric và ngưỡng cải thiện tối thiểu để lên champion nằm trong `configs/promotion/default.yaml`
(`promotion.metric`, `promotion.min_delta`); `retrain.retrain(cfg)` là điểm vào cho DAG `retrain`.

`decision_threshold` và `input_schema` nằm trong `model_card.json` của run nguồn
(`MlflowClient().download_artifacts(version.run_id, "model_card.json")`). API chưa đọc từ Registry; khi nối vào API
(`MODEL_ALIAS`) cần PR riêng của Serving.
