# Notebook

Các notebook phân tích gốc của môn IT3190 (Nhập môn Học máy và Khai phá dữ liệu), nơi dữ liệu được khảo sát và
mô hình champion được chọn. Chúng được giữ nguyên code và kết quả chạy (bảng, hình) để làm tài liệu tham khảo cho báo
cáo; đường dẫn tuyệt đối trong output đã được đổi thành đường dẫn tương đối.

| Notebook | Nội dung |
|---|---|
| `01_data_understanding_and_eda.ipynb` | Tìm hiểu dữ liệu, EDA, các cột rò rỉ nhãn |
| `02_data_preprocessing.ipynb` | Làm sạch, chia dữ liệu theo thời gian, đặc tả đặc trưng |
| `03_model_training.ipynb` | CV theo thời gian, so sánh 10 mô hình, tinh chỉnh, chọn ensemble champion |
| `full_pipeline.ipynb` | Pipeline đầy đủ của mô hình tốt nhất, từ dữ liệu thô tới model card |

**Không chạy trực tiếp được trong repo này**: notebook dùng cấu trúc thư mục và tên module của project IT3190. Phần
code cần cho vận hành đã được chuyển vào `src/hotel_booking/`:

| Trong notebook | Trong repo này |
|---|---|
| Thư mục dữ liệu `dataset/raw`, `dataset/processed` | `data/raw` (DVC), `data/processed` (`uv run dvc repro`) |
| `hotel_booking.config`, `data`, `features` | Cùng tên |
| `hotel_booking.modeling` | `hotel_booking.model` |
| `hotel_booking.evaluation` | `hotel_booking.evaluate` |
| `hotel_booking.experiments` (CV out-of-fold, random search, ensemble) | Không chuyển; tinh chỉnh bằng Optuna + MLflow ở `hotel_booking.tune` |
| Spec và kết quả champion | `configs/model/champion.yaml`, `reports/it3190_model_card.json` |

Champion train bằng code của repo cho ra cùng kết quả trên test 01–04/2017 (sai lệch không quá 0,005), kiểm tra bởi
`tests/test_it3190_reproduction.py`.
