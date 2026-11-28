# Báo cáo cuối kỳ: Dự đoán hủy đặt phòng khách sạn — hệ thống MLOps

Khung báo cáo (task 2.1.1). Mỗi mục ghi người viết, các ý cần có và tư liệu đã có sẵn trong repo. Người viết thay
phần gạch đầu dòng bằng nội dung, giữ nguyên thứ tự mục. Hình đặt trong `reports/figures/`, chèn bằng đường dẫn
tương đối. Task 2.1.3 (đối chiếu checklist) và 2.1.4 (review chéo) dùng chính file này.

| Mục | Người viết | Review chéo |
|---|---|---|
| 1. Giới thiệu, 4. Kiến trúc, 9. CI/CD, 11. Đối chiếu checklist | Quân | Đức |
| 2. Dữ liệu, 7.1 Drift | Nhật Anh | Tùng |
| 3. Mô hình và huấn luyện | Tùng | Nhật Anh |
| 5. Serving | Đức | Quang |
| 6. Frontend và chất lượng | Quang | Quân |
| 7.2 Prometheus và Grafana, 8. Huấn luyện liên tục | Quân (8.2: Tùng, 8.3: Nhật Anh) | Đức |
| 10. Kết quả demo, 12. Kết luận | Cả nhóm (Quân tổng hợp) | — |

## 1. Giới thiệu

*Người viết: Quân*

- Bài toán: dự đoán một đặt phòng có bị hủy hay không tại thời điểm xác nhận; giá trị cho khách sạn (overbooking,
  chính sách đặt cọc).
- Mục tiêu môn học: vận hành mô hình như một hệ thống — tái lập, tự động hóa, triển khai, giám sát, huấn luyện lại.
- Phạm vi: chạy local bằng docker compose, không triển khai cloud; dịch vụ dùng chung DagsHub.
- Tư liệu: `README.md` mục 1–2.

## 2. Dữ liệu

*Người viết: Nhật Anh*

- 2.1 Nguồn và mô tả: Hotel booking demand (Kaggle), 119 390 dòng, 2015-07 → 2017-08, City Hotel và Resort Hotel.
- 2.2 Làm sạch và chống rò rỉ nhãn: bỏ `reservation_status*`, xử lý giá trị thiếu, 185 dòng bị loại (số liệu trong
  `tests/test_data.py`).
- 2.3 Chia theo thời gian: train 07/2015–12/2016 (78 586), test 01–04/2017 (18 442), production holdout 05–08/2017
  (22 177) dùng để replay.
- 2.4 Version dữ liệu: DVC pipeline `prepare → validate → drift`, remote DagsHub và SeaweedFS.
- 2.5 Kiểm tra dữ liệu: schema pandera theo từng split (`src/hotel_booking/validation.py`,
  `reports/data_validation.json`).
- Tư liệu: `notebooks/01_data_understanding_and_eda.ipynb`, `notebooks/02_data_preprocessing.ipynb`, `dvc.yaml`.
- Hình cần có: phân bố nhãn theo tháng, sơ đồ chia dữ liệu theo thời gian.

## 3. Mô hình và huấn luyện

*Người viết: Tùng*

- 3.1 Mô hình champion từ IT3190: ensemble Random Forest + XGBoost + LightGBM, trọng số ưu tiên dữ liệu gần; CV theo
  quý, ROC-AUC test 0,865 (`reports/it3190_model_card.json`, `notebooks/03_model_training.ipynb`).
- 3.2 Chuyển sang repo: `build_from_spec`, kết quả tái lập trong 0,005 (`tests/test_it3190_reproduction.py`).
- 3.3 Pipeline huấn luyện: Hydra config, `train.py`/`evaluate.py`, seed cố định.
- 3.4 Theo dõi thí nghiệm: MLflow (params, metrics, artifact, model card), Optuna lồng trong MLflow.
- 3.5 Model Registry: alias `champion`/`challenger`, `retrain()` và cổng so sánh `promotion.min_delta`.
- 3.6 Profiling huấn luyện và tiền xử lý (Quang viết đoạn này): `reports/profiling.md`.
- Hình cần có: ảnh chụp MLflow UI (danh sách run, so sánh, Registry), bảng kết quả Optuna.

## 4. Kiến trúc hệ thống

*Người viết: Quân*

- Sơ đồ: `reports/figures/architecture.png` (bản sửa: `architecture.excalidraw`).
- Danh sách service trong docker compose, cổng, cách các service gọi nhau: `docs/source/contracts.md` C11.
- Các "hợp đồng" giữa module (C1–C14) và lý do cần chúng khi 5 người làm song song.
- Quyết định đáng nêu: DagsHub để dùng chung, SeaweedFS thay MinIO (MinIO ngừng phát hành image), Airflow không cài
  dependency dự án.

## 5. Serving

*Người viết: Đức*

- 5.1 FastAPI: `/predict`, `/predict/batch`, schema pydantic, cảnh báo giá trị lạ (C1).
- 5.2 Nạp mô hình: từ file có checksum hoặc từ MLflow Registry (`MODEL_SOURCE=mlflow`), `POST /reload` không cần
  khởi động lại (C6, C7).
- 5.3 ONNX: converter tự viết, kiểm tra khớp sklearn, benchmark latency/throughput/kích thước
  (`reports/serving_benchmark.md`).
- 5.4 Log dự đoán vào PostgreSQL bằng background task, nhãn trễ qua replay (C9).
- 5.5 Docker image và GHCR.

## 6. Frontend và chất lượng

*Người viết: Quang*

- 6.1 Streamlit: form một booking, upload CSV, xử lý lỗi khi API không phản hồi.
- 6.2 Độ bền mô hình: ROC-AUC theo tháng 2017 và khi đổi khách sạn (`reports/durability.md`,
  `reports/figures/durability_*.png`).
- 6.3 Load test Locust: joblib so với ONNX qua HTTP (task 1.5.9).
- 6.4 Workflow khi dữ liệu thay đổi.
- Hình cần có: ảnh chụp giao diện Streamlit.

## 7. Giám sát

### 7.1 Data drift

*Người viết: Nhật Anh*

- Báo cáo Evidently train so với từng tháng 2017 và City → Resort Hotel (`reports/drift.md`).
- Drift service đọc log dự đoán, endpoint `/drift`, metrics cho Prometheus (C14).

### 7.2 Prometheus, Grafana và cảnh báo

*Người viết: Quân*

- Metrics của API (C10) và của drift service; Prometheus scrape, 5 luật cảnh báo (`monitoring/prometheus/alerts.yml`).
- Dashboard `monitoring/grafana/dashboards/hotel-booking.json`.
- Hình cần có: dashboard khi replay, cảnh báo `DataDrift` và `ApiDown` đang bật.

## 8. Huấn luyện liên tục

- 8.1 Airflow 3 trong docker compose, kiểm tra DAG trên CI (Quân).
- 8.2 DAG `retrain` (Tùng).
- 8.3 DAG `monitor_drift`: ngưỡng `drift_share_threshold`, mỗi tháng kích hoạt tối đa một lần (Nhật Anh).
- Hình cần có: Airflow UI với một lần `retrain` do `monitor_drift` kích hoạt.

## 9. CI/CD

*Người viết: Quân*

- Workflow: Unit Tests (Ubuntu, Windows, macOS), Code linting, Data changed, CML, Airflow DAGs, Docker images → GHCR.
- Quy trình nhánh, pre-commit, Dependabot (`CONTRIBUTING.md`).
- Hình cần có: một comment CML trên PR, danh sách workflow xanh.

## 10. Kết quả demo

*Cả nhóm, Quân tổng hợp*

- Kịch bản vòng khép kín (task 1.7.4): replay 2017 → drift → retrain → promote → API dùng champion mới.
- Số liệu thực tế: ROC-AUC trước và sau retrain, `drift_share` từng tháng, thời gian một vòng.

## 11. Đối chiếu yêu cầu và checklist

*Người viết: Quân (task 2.1.3)*

- Bảng yêu cầu → nơi đáp ứng trong repo; mục không áp dụng và lý do (huấn luyện phân tán, PyTorch Lightning).

## 12. Kết luận

*Cả nhóm*

- Kết quả đạt được, hạn chế, hướng phát triển.

## Phụ lục

- Phân công: `README.md` mục 8.
- Cách chạy lại toàn bộ: `README.md` mục 5.
