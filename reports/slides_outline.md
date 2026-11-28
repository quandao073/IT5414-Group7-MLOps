# Outline slide thuyết trình

Outline và phân công (task 2.2.1). Mỗi người làm slide phần mình (2.2.2), Quân ghép và chạy tổng duyệt (2.2.3). Thời
lượng giả định 20 phút trình bày + demo; sửa theo lịch lớp. Nội dung lấy từ `reports/report.md` cùng mục.

| # | Slide | Người làm | Thời gian | Nội dung chính | Hình / tư liệu |
|---|---|---|---|---|---|
| 1 | Tiêu đề, thành viên | Quân | 0,5 phút | Tên đề tài, 5 thành viên và vai trò | `README.md` mục 8 |
| 2 | Bài toán và mục tiêu | Quân | 1 phút | Dự đoán hủy phòng; vì sao cần MLOps chứ không chỉ một notebook | `README.md` mục 1 |
| 3 | Kiến trúc tổng thể | Quân | 2 phút | Đi qua 6 khu vực của sơ đồ | `reports/figures/architecture.png` |
| 4 | Dữ liệu và DVC | Nhật Anh | 1,5 phút | Chia theo thời gian, chống rò rỉ, pipeline DVC, pandera | notebook 01, 02; `dvc.yaml` |
| 5 | Mô hình và huấn luyện | Tùng | 2 phút | Champion IT3190 (ROC-AUC 0,865), Hydra, Optuna, MLflow | ảnh MLflow UI |
| 6 | Model Registry và retrain | Tùng | 1 phút | `champion`/`challenger`, cổng promote | ảnh Registry |
| 7 | Serving | Đức | 2 phút | API, `/reload`, ONNX và benchmark, log dự đoán | `reports/serving_benchmark.md` |
| 8 | Frontend và chất lượng | Quang | 1,5 phút | Streamlit, profiling, độ bền mô hình, load test | `reports/figures/durability_*.png`, `profile_*.png` |
| 9 | Drift | Nhật Anh | 1,5 phút | Drift từng tháng 2017, City → Resort, drift service | `reports/drift.md` |
| 10 | Giám sát | Quân | 1 phút | Prometheus, Grafana, 5 cảnh báo | ảnh dashboard, cảnh báo đang bật |
| 11 | CI/CD | Quân | 1 phút | 6 workflow, CML comment trên PR, GHCR | ảnh Actions, comment CML |
| 12 | Demo live: vòng khép kín | Đức + Tùng | 5 phút | Replay → drift → `monitor_drift` → `retrain` → promote → `/reload` → Grafana | video dự phòng 3–5 phút |
| 13 | Kết quả và đối chiếu yêu cầu | Quân | 1 phút | Checklist đã đáp ứng, mục không áp dụng | `reports/report.md` mục 11 |
| 14 | Hạn chế và hướng phát triển, hỏi đáp | Cả nhóm | 1 phút | | |

**Chuẩn bị demo (task 1.7.4):**

- Máy demo chạy sẵn `docker compose up -d`, đã có model champion và dữ liệu `data/processed`.
- Bật DAG `monitor_drift`, đặt lịch ngắn (`MONITOR_DRIFT_SCHEDULE`) để thấy kích hoạt trong vài phút.
- Mở sẵn các tab: Streamlit :8501, Grafana :3000, Airflow :8080, MLflow :5000, Prometheus :9090/alerts.
- Quay video dự phòng trước buổi thuyết trình (2.2.2).
