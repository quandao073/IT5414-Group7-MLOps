# Data drift: dữ liệu train so với từng tháng 2017 (task 1.6.4)

Sinh bằng `uv run dvc repro drift` (hoặc `uv run invoke drift-report`) từ `data/processed`. Số liệu dưới đây lấy từ
`reports/drift/summary.csv`; chi tiết từng cột ở `reports/drift/columns.csv`, report HTML của Evidently ở
`reports/drift/html/<kịch bản>.html` (quản lý bằng DVC, lấy về bằng `uv run dvc pull`).

## Cách đo

- **Tham chiếu:** toàn bộ tập train (đặt phòng đến từ 07/2015 tới 12/2016, 78.586 dòng).
- **Hiện tại:** từng tháng đến của 2017 (01–04 thuộc `test`, 05–08 thuộc `production_holdout`), đúng các batch
  mà replay phát vào API.
- **Cột so sánh (22):** đầu vào của mô hình, bỏ 4 cột lịch (năm, tháng, tuần, ngày) vì chúng khác theo định nghĩa khi
  so hai giai đoạn; `agent`, `company` là mã định danh nên so như biến phân loại.
- **Kiểm định:** Evidently `DataDriftPreset` tự chọn theo kiểu cột và cỡ mẫu. Với hơn 1.000 dòng: khoảng cách
  Wasserstein (chuẩn hóa) cho cột số và Jensen–Shannon cho cột phân loại, ngưỡng 0,1. Tập dữ liệu được coi là drift
  khi ít nhất 50% số cột bị drift.

## Kết quả

| Kịch bản | Dòng | Cột drift | Tỉ lệ | Drift tập dữ liệu |
|---|---:|---:|---:|:---:|
| 2017-01 | 3.673 | 14/22 | 0,64 | có |
| 2017-02 | 4.168 | 11/22 | 0,50 | có |
| 2017-03 | 4.948 | 9/22 | 0,41 | không |
| 2017-04 | 5.653 | 11/22 | 0,50 | có |
| 2017-05 | 6.305 | 9/22 | 0,41 | không |
| 2017-06 | 5.639 | 10/22 | 0,45 | không |
| 2017-07 | 5.310 | 16/22 | 0,73 | có |
| 2017-08 | 4.923 | 17/22 | 0,77 | có |
| City → Resort Hotel (trong train) | 26.872 | 15/21 | 0,71 | có |

## Nhận xét

- **Cột luôn drift:** `adr`, `lead_time`, `agent`, `company`, `days_in_waiting_list` drift ở cả 8 tháng. `adr` (giá
  phòng trung bình) tăng theo năm và theo mùa, nên một tháng đơn lẻ luôn khác phân phối của cả 18 tháng train.
- **Mùa vụ:** tháng 7 và 8/2017 drift mạnh nhất (16–17/22 cột): ngoài các cột trên còn có số người lớn (và trẻ em
  ở tháng 8), số đêm lưu trú, loại bữa ăn, loại phòng đặt và loại đặt cọc, đúng đặc điểm mùa du lịch hè. Riêng cột
  `hotel` không drift, nên tỉ lệ hai khách sạn gần như giữ nguyên. Tháng 3, 5, 6 gần với train nhất.
- **Đổi khách sạn:** City Hotel và Resort Hotel khác nhau ở 15/21 cột ngay trong cùng giai đoạn train, mạnh hơn hầu hết
  các tháng 2017. Đây là kịch bản kiểm tra độ bền của mô hình khi đổi khách sạn (đánh giá ROC-AUC ở task 1.6.5).
- **Hệ quả cho giám sát:** với tham chiếu là cả tập train, drift theo tháng phần lớn là mùa vụ chứ chưa chắc làm mô hình
  kém đi. Vì vậy ngưỡng kích hoạt retrain trong DAG `monitor_drift` để thành Airflow Variable
  (`drift_share_threshold`, mặc định 0,5), và quyết định cuối cùng vẫn là cổng champion–challenger so trên tháng có
  nhãn (task 1.3.8).
