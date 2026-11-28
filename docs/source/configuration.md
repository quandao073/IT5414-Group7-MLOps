# Cấu hình

Dự án tách cấu hình huấn luyện (Hydra) khỏi cấu hình runtime (biến môi trường). Trang này
liệt kê cả hai.

## Hydra

Entry point `configs/config.yaml` gộp các nhóm config:

```yaml
defaults:
  - model: lgbm
  - data: default
  - train: default
  - promotion: default
  - _self_
```

| Nhóm | Thư mục | Nội dung |
|---|---|---|
| Model | `configs/model/` | Một file cho mỗi mô hình hoặc ensemble: `lgbm`, `logreg`, `logistic_regression`, `champion` |
| Data | `configs/data/` | Đường dẫn dữ liệu đã xử lý |
| Train | `configs/train/` | Seed, thư mục output, experiment MLflow, bật/tắt tracking và registry |
| HPO | `configs/hpo/` | Không gian tìm kiếm Optuna |
| Promotion | `configs/promotion/` | Metric và `min_delta` để promote challenger |

Chọn mô hình hoặc đổi siêu tham số bằng override trên dòng lệnh, không sửa code:

```bash
uv run invoke train --overrides "model=logreg train.seed=7 train.register=true"
uv run invoke train --overrides "model=champion train.tracking=false"
uv run invoke tune --overrides "hpo.n_trials=10"
```

Các giá trị mặc định đáng chú ý trong `configs/config.yaml`:

| Khóa | Mặc định | Ý nghĩa |
|---|---|---|
| `seed` | `42` | Seed cho chia dữ liệu và mô hình |
| `output_dir` | `models` | Nơi ghi `model.joblib` và `model_card.json` |
| `experiment` | `hotel-booking-cancellation` | Tên experiment trên MLflow |
| `tracking` | `true` | Log run lên MLflow |
| `register` | `false` | Đăng ký mô hình vào MLflow Registry |
| `processed_dir` | `data/processed` | Dữ liệu đầu vào của train |

## Biến môi trường

Các biến cho máy local nằm trong `.env.example`; sao chép thành `.env` (file này không được
commit). Bảng dưới bổ sung các biến chỉ dùng trong compose hoặc Airflow. Hợp đồng C5 trong
[Hợp đồng](contracts.md#c5-bien-moi-truong) là nguồn chuẩn.

| Biến | Mặc định | Dùng bởi |
|---|---|---|
| `MODEL_SOURCE` | `file` | API |
| `MODEL_BACKEND` | `joblib`; `onnx` để chạy ONNX Runtime | API |
| `MODEL_PATH` | `models/model.joblib` | API |
| `MODEL_CARD_PATH` | `models/model_card.json` | API |
| `RELOAD_TOKEN` | trống — `/reload` không cần token | API |
| `API_URL` | `http://localhost:8000` | Replay, frontend, load test |
| `MLFLOW_TRACKING_URI` | MLflow trên DagsHub; tự host: `http://localhost:5000` | Training, API |
| `MLFLOW_TRACKING_USERNAME` | Tên tài khoản DagsHub | Training, API |
| `MLFLOW_TRACKING_PASSWORD` | Access token DagsHub (bí mật) | Training, API |
| `CONFIG_DIR` | `configs/` | Training |
| `POSTGRES_USER`, `POSTGRES_PASSWORD` | `mlops`, `mlops` (local) | docker compose |
| `S3_ACCESS_KEY`, `S3_SECRET_KEY` | `mlops`, `mlops-secret` (local) | SeaweedFS, MLflow, DVC remote `local` |
| `AIRFLOW_JWT_SECRET` | `local-only-jwt-secret` (local) | docker compose |
| `MONITOR_DRIFT_SCHEDULE` | `*/10 * * * *`; đặt ngắn hơn khi demo | DAG `monitor_drift` |
| `GRAFANA_ADMIN_PASSWORD` | `admin` (local) | Grafana |
| `DATABASE_URL` | không có — tắt log dự đoán | API, replay, drift service |
| `REFERENCE_DIR` | `data/processed` | Drift service |
| `DRIFT_MIN_ROWS` | `100` | Drift service |
| `DRIFT_SERVICE_URL` | `http://drift:8001` | DAG `monitor_drift` |

!!! warning "Bí mật"
    `MLFLOW_TRACKING_PASSWORD`, `RELOAD_TOKEN`, mật khẩu PostgreSQL và khóa S3 chỉ đặt trong
    `.env` hoặc secret của CI. Không hard-code trong code hay config được commit.

## MLflow và DVC

### DagsHub (mặc định khi phát triển)

```dotenv
MLFLOW_TRACKING_URI=https://dagshub.com/quandao073/IT5414-Group7-MLOps.mlflow
MLFLOW_TRACKING_USERNAME=<tên-tài-khoản>
MLFLOW_TRACKING_PASSWORD=<access-token>
```

DVC dùng cùng tài khoản, lưu trong `.dvc/config.local`:

```bash
uv run dvc remote modify origin --local user <tên-tài-khoản>
uv run dvc remote modify origin --local password <access-token>
```

Quy trình tạo token và cấu hình lần đầu xem [Bắt đầu](getting-started.md#cau-hinh-dagshub).

### MLflow tự host

Khi chạy `docker compose`, MLflow chạy trong mạng compose và không cần tài khoản:

```dotenv
MLFLOW_TRACKING_URI=http://localhost:5000
```

Bên trong container, dùng `http://mlflow:5000`. Artifact lưu trên SeaweedFS (S3).

## Backend suy luận

API phục vụ cùng một mô hình qua hai backend, chọn bằng `MODEL_BACKEND`:

| Giá trị | Cách chạy | Khi nào dùng |
|---|---|---|
| `joblib` | scikit-learn, nạp `model.joblib` | Mặc định |
| `onnx` | ONNX Runtime, nạp `model.onnx` | Đo độ trễ và kích thước artifact |

```bash
uv run invoke export-onnx
uv run invoke benchmark
```

Request, response và `/reload` giống nhau ở cả hai backend; chỉ metric `api_model_info` và
cột `backend` trong log dự đoán khác. Chi tiết xem [Hợp đồng](contracts.md#c8-backend-onnx).

## Cấu hình docker compose

Các biến `POSTGRES_*`, `S3_*` và `AIRFLOW_JWT_SECRET` được `docker-compose.yml` đọc trực tiếp
từ `.env`. Giá trị mặc định chỉ dành cho máy local; không dùng lại khi triển khai thật.
