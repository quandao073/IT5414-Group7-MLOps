# Benchmark suy luận: joblib và ONNX

Tài liệu này ghi lại kết quả của task **1.4.5**: so sánh độ trễ, thông lượng và kích thước artifact của cùng một mô
hình khi API chạy bằng scikit-learn (`MODEL_BACKEND=joblib`) và bằng ONNX Runtime (`MODEL_BACKEND=onnx`). Cách export
và cách API nạp ONNX: contract C8 trong `docs/source/contracts.md`.

## Cách chạy lại

```bash
uv run dvc pull
uv run invoke preprocess-data
uv run src/hotel_booking/sample_model.py --spec-path configs/model/champion.yaml --output-dir models/bench_champion
uv run src/hotel_booking/onnx_export.py --model-dir models/bench_champion
uv run src/hotel_booking/benchmark.py --model-dir models/bench_champion --output reports/benchmark/champion.json
```

Đổi `champion` thành `logistic_regression` để đo baseline. Số đo gốc: `reports/benchmark/*.json`.

## Môi trường đo

- Laptop Windows 11, Intel Core thế hệ 11 (8 luồng), chỉ CPU; Python 3.12.13.
- pandas 3.0.6, numpy 2.5.5, scikit-learn 1.9.1, LightGBM 4.7.0, XGBoost 3.4.1, onnx 1.23.2, onnxruntime 1.31.1,
  skl2onnx 1.20.0, onnxmltools 1.16.0.
- Dữ liệu: `test.csv` thật (01–04/2017), 18 442 dòng. Đo 27/11/2026.
- Thời gian là một lần gọi `predict_proba`: tạo đặc trưng bằng pandas + suy luận, **không gồm HTTP**. Đo bằng
  `time.perf_counter()`, có 3 lần gọi khởi động không tính. 1 bản ghi: 300 lần gọi; lô 1000 bản ghi: 30 lần gọi.
- Hai backend được đo liền nhau trong cùng một lần chạy, trên cùng mô hình và cùng dữ liệu, nên so sánh được với nhau.
  Số đo ở các thời điểm khác nhau có thể lệch nhiều (xem `reports/profiling.md`).

## Mô hình được đo

Train bằng `sample-model` trên train thật (07/2015–12/2016); ngưỡng chọn theo F1 trên 3 tháng cuối của train.

| Mô hình | Spec | Test ROC-AUC | Test PR-AUC | Test F1 |
|---|---|---:|---:|---:|
| Logistic Regression | `configs/model/logistic_regression.yaml` | 0,859 | 0,824 | 0,682 |
| Champion (RF + XGBoost + LightGBM) | `configs/model/champion.yaml` | 0,865 | 0,824 | 0,694 |

ROC-AUC và PR-AUC của champion khớp `reports/it3190_model_card.json` (0,865 / 0,824).

## Kết quả

| Mô hình | Backend | 1 bản ghi p50 (ms) | 1 bản ghi p95 (ms) | Lô 1000 p50 (ms) | Bản ghi/giây (lô) | Artifact (MB) |
|---|---|---:|---:|---:|---:|---:|
| Logistic Regression | joblib | 28,8 | 42,3 | 41,8 | 23 945 | 0,027 |
| Logistic Regression | ONNX | 19,1 | 26,0 | 30,1 | 33 207 | 0,015 |
| Champion | joblib | 169,7 | 312,7 | 231,8 | 4 314 | 70,4 |
| Champion | ONNX | 110,9 | 168,5 | 142,7 | 7 007 | 34,7 |

Độ lệch xác suất ONNX so với scikit-learn trên 5000 dòng test: tối đa 1,42·10⁻⁷ (Logistic Regression) và
2,84·10⁻⁷ (champion), nên hai backend cho cùng quyết định.

## Nhận xét

- **Champion:** ONNX giảm độ trễ 1 bản ghi khoảng 35 % (p50 170 → 111 ms) và gần một nửa ở p95 (313 → 168 ms), tăng
  thông lượng theo lô khoảng 1,6 lần (4 314 → 7 007 bản ghi/giây) và giảm một nửa kích thước artifact (70,4 → 34,7 MB).
  Phần lớn kích thước là Random Forest (`model.0.onnx` 34,0 MB).
- **Logistic Regression:** mô hình rất nhẹ nên thời gian chủ yếu nằm ở bước tạo đặc trưng bằng pandas, bước này chạy
  bằng Python ở cả hai backend; ONNX chỉ nhanh hơn khoảng 1/3.
- Backend ONNX vẫn nạp `model.joblib` để lấy bộ tạo đặc trưng, nên dung lượng trên đĩa và bộ nhớ khi chạy không giảm
  theo cột Artifact; cột này là kích thước phần suy luận.
- **Đánh đổi:** ONNX cần converter riêng cho bước `log1p_nonnegative` và `OneHotEncoder` có category hiếm; khi bước tiền
  xử lý trong `model.py` / `features.py` đổi thì phải cập nhật converter (test `tests/test_onnx.py` sẽ báo nếu lệch).
- Độ trễ qua HTTP (FastAPI, kiểm tra pydantic, ghi log) cao hơn các số trên; load test 1.5.9 đo phần đó.
