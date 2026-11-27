# Bắt đầu

Trang này hướng dẫn cài đặt môi trường phát triển, lấy dữ liệu và chạy các thành phần
chính của dự án trên máy cá nhân.

## Yêu cầu

- [Git](https://git-scm.com/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — quản lý Python 3.12 và dependency
- Docker Desktop — chỉ cần khi chạy toàn bộ hệ thống bằng `docker compose`
- Tài khoản [DagsHub](https://dagshub.com/) được thêm làm collaborator — để `dvc pull` và log MLflow

## Cài đặt môi trường

```bash
git clone https://github.com/quandao073/IT5414-Group7-MLOps.git
cd IT5414-Group7-MLOps
git switch develop
uv sync
uv run pre-commit install
```

`uv sync` tạo `.venv` với Python 3.12 theo `uv.lock`. `pre-commit install` cài hook chạy
`ruff` và kiểm tra file trước mỗi commit.

Trên macOS, XGBoost và LightGBM cần OpenMP:

```bash
brew install libomp
```

## Cấu hình DagsHub

MLflow tracking và DVC remote dùng chung repo DagsHub
[`quandao073/IT5414-Group7-MLOps`](https://dagshub.com/quandao073/IT5414-Group7-MLOps).
Làm một lần trên mỗi máy, trước `dvc pull`:

1. Tạo access token tại [dagshub.com/user/settings/tokens](https://dagshub.com/user/settings/tokens).
2. Sao chép `.env.example` thành `.env` và điền:

   ```dotenv
   MLFLOW_TRACKING_URI=https://dagshub.com/quandao073/IT5414-Group7-MLOps.mlflow
   MLFLOW_TRACKING_USERNAME=<tên-tài-khoản>
   MLFLOW_TRACKING_PASSWORD=<access-token>
   ```

3. Lưu tài khoản cho DVC vào `.dvc/config.local` (file không được commit):

   ```bash
   uv run dvc remote modify origin --local user <tên-tài-khoản>
   uv run dvc remote modify origin --local password <access-token>
   ```

!!! warning "Không commit bí mật"
    `.env` và `.dvc/config.local` đã nằm trong `.gitignore`. Token chỉ đặt ở đó, không dán
    vào issue, PR hay chat.

## Lấy dữ liệu và chạy pipeline

```bash
uv run dvc pull
uv run dvc repro
```

`dvc repro` chạy ba stage:

| Stage | Việc làm | Output |
|---|---|---|
| `prepare` | Làm sạch, chia theo thời gian, tạo schema | `data/processed/` |
| `validate` | Kiểm tra từng split bằng pandera | `reports/data_validation.json` |
| `drift` | So sánh train với từng tháng 2017 và giữa hai khách sạn | `reports/drift/` |

Từng stage cũng chạy riêng được:

```bash
uv run invoke preprocess-data
uv run invoke validate-data
uv run invoke drift-report
```

Dữ liệu đã xử lý không commit; nó được tạo lại từ dữ liệu thô bằng các lệnh trên.

## Chạy test

```bash
uv run invoke test
```

Unit test chạy được cả khi chưa có dữ liệu thật: các test cần file sẽ tự bỏ qua.

## Huấn luyện

```bash
uv run invoke train
uv run invoke evaluate --split test
```

`train` đọc `configs/model/<model>.yaml`, log params, metrics, confusion matrix và feature
importance lên MLflow, rồi lưu `models/model.joblib` cùng `models/model_card.json`.
Override cấu hình bằng Hydra:

```bash
uv run invoke train --overrides "model=logreg train.seed=7 train.register=true"
```

Các lệnh liên quan:

```bash
uv run invoke tune --overrides "hpo.n_trials=10"
uv run invoke retrain --overrides "promotion.min_delta=0.002"
uv run invoke registry --command "show"
```

## Phục vụ mô hình

```bash
uv run invoke sample-model
uv run invoke serve-api
```

- API chạy tại `http://localhost:8000`, Swagger UI tại `/docs`.
- `POST /reload` nạp lại mô hình sau khi ghi file mới, không cần restart.
- `uv run invoke serve-frontend` mở Streamlit tại `http://localhost:8501` (cần API đang chạy).

Các lệnh serving khác:

```bash
uv run invoke export-onnx
uv run invoke benchmark
uv run invoke replay --months 2017-01,2017-02 --delay 0.5
```

## Chạy bằng Docker compose

```bash
docker compose up -d --build
docker compose ps
docker compose down
```

Các service: API (`:8000`), Streamlit (`:8501`), MLflow (`:5000`), Airflow (`:8080`),
PostgreSQL (`:5432`), SeaweedFS S3 (`:8333`), Prometheus và Grafana. Chi tiết cổng và biến
môi trường xem [Hợp đồng](contracts.md) mục C11.

## Lệnh invoke thường dùng

| Lệnh | Việc làm |
|---|---|
| `uv run invoke test` | Chạy test kèm coverage |
| `uv run invoke preprocess-data` | Sinh `data/processed/` |
| `uv run invoke validate-data` | Validate schema bằng pandera |
| `uv run invoke drift-report` | Sinh báo cáo drift |
| `uv run invoke train` | Huấn luyện theo Hydra config |
| `uv run invoke evaluate` | Đánh giá mô hình đã lưu |
| `uv run invoke tune` | Tối ưu LightGBM bằng Optuna |
| `uv run invoke retrain` | Train + promote champion/challenger |
| `uv run invoke sample-model` | Tạo mô hình mẫu cho API |
| `uv run invoke serve-api` | Chạy FastAPI |
| `uv run invoke serve-frontend` | Chạy Streamlit |
| `uv run invoke serve-drift` | Chạy drift service |
| `uv run invoke build-docs` | Build tài liệu (`--strict`) |
| `uv run invoke serve-docs` | Xem tài liệu tại `http://127.0.0.1:8000` |

## Xử lý sự cố

- **PowerShell báo exit code 1 dù lệnh thành công**: PowerShell 5.1 trả mã lỗi khi tool
  native ghi cảnh báo ra stderr (DVC, MLflow). Đọc nội dung output thay vì chỉ nhìn mã thoát.
- **`dvc pull` trả 401**: kiểm tra lại `user`/`password` trong `.dvc/config.local` và token
  DagsHub còn hiệu lực.
- **`uv` báo fallback hardlink**: cache và ổ đĩa khác filesystem; đặt `UV_LINK_MODE=copy`
  nếu cần.
