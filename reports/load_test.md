# Load test API: joblib vs ONNX

Kết quả task **1.5.9** — đo API dự đoán qua HTTP bằng Locust với hai backend của cùng một mô hình.

## Mục tiêu

So sánh `POST /predict` khi API phục vụ `models/model.joblib` (backend `joblib`) và `models/model.onnx` (backend `onnx`): độ trễ p50/p95, số request mỗi giây (RPS) và tỉ lệ lỗi.

## Mô hình và môi trường

| Mục | Giá trị |
|---|---|
| Mô hình | Logistic Regression — `logistic_regression-20261113144119` |
| Ngưỡng quyết định | `0.18254583193161286` |
| ONNX | opset 17, kiểm tra parity trên 5 000 dòng, lệch tuyệt đối lớn nhất `1.42e-07` |
| Máy chạy | Windows 11 Home 10.0.26300, AMD Ryzen 7 5800H (8 nhân / 16 luồng), 15.9 GB RAM |
| Python | 3.12.15 |
| API | 1 uvicorn worker; Locust chạy **cùng máy** với API |

## Profile tải

- 20 người dùng ảo, spawn rate 5 người/giây, chạy 60 giây.
- Mỗi request gửi một booking lấy từ `docs/examples/predict_request.json` tới `POST /predict`.
- Thời gian chờ giữa hai request của mỗi người dùng: 10–50 ms.
- Không đo `POST /predict/batch`; đây là so sánh một request một booking.

## Cách chạy lại

```bash
uv run invoke export-onnx
```

Cửa sổ 1 — API với joblib:

```powershell
$env:MODEL_BACKEND="joblib"; uv run invoke serve-api
```

Cửa sổ 2 — đo (lệnh tự chờ `/health` sẵn sàng và kiểm tra `/metrics` đang phục vụ đúng backend):

```bash
uv run invoke load-test --backend joblib --csv-prefix reports/load_test/joblib
```

Dừng API, lặp lại với backend ONNX:

```powershell
$env:MODEL_BACKEND="onnx"; uv run invoke serve-api
```

```bash
uv run invoke load-test --backend onnx --csv-prefix reports/load_test/onnx
```

Sau đó tổng hợp:

```bash
uv run python tests/performance/summarize.py reports/load_test/joblib_stats.csv reports/load_test/onnx_stats.csv
```

## Kết quả

| Backend | Requests | RPS | p50 (ms) | p95 (ms) | Error rate |
|---|---:|---:|---:|---:|---:|
| joblib | 683 | 11.46 | 1600.0 | 1900.0 | 0.00% |
| onnx | 936 | 15.78 | 1200.0 | 1400.0 | 0.00% |

Cả hai lượt đều có 0 request lỗi; `joblib_exceptions.csv` và `onnx_exceptions.csv` chỉ có dòng tiêu đề, tức không có lỗi kết nối bị ẩn.

### Kiểm chứng backend

`invoke load-test --backend ...` tự chờ `/health` sẵn sàng, đọc `/metrics` trước khi chạy và dừng nếu backend không khớp. Kiểm chứng lại bằng cách khởi động API lần lượt với từng backend, `/metrics` trả về:

```text
api_model_info{backend="joblib",model_version="logistic_regression-20261113144119"} 1.0
api_model_info{backend="onnx",model_version="logistic_regression-20261113144119"} 1.0
```

`/health` của cả hai lượt trả `model_loaded: true`, cùng `model_version`.

## Nhận xét

- ONNX đạt RPS cao hơn khoảng 37.7% (15.78 so với 11.46) và giảm p50/p95 lần lượt khoảng 25%/26%.
- Mức cải thiện không lớn hơn nữa vì phần tạo đặc trưng bằng pandas và xử lý request trong Python chiếm phần lớn thời gian; ONNX chỉ tăng tốc bước suy luận của mô hình.
- Độ trễ tuyệt đối (1.2–1.9 giây) cao hơn nhiều so với benchmark trong tiến trình vì có thêm HTTP, validate Pydantic và tranh chấp CPU giữa Locust với API.

## Lưu ý

- API và Locust chạy cùng một máy nên số đo chỉ dùng để so sánh tương đối giữa hai backend, không phải năng lực production.
- Chỉ một uvicorn worker và không batch; cấu hình nhiều worker hoặc batch sẽ cho kết quả khác.
- Số liệu phụ thuộc phần cứng; chạy lại trên máy khác trước khi trích dẫn.
- Bảng lấy từ `*_stats.csv` do Locust ghi khi kết thúc; tổng request có thể lệch vài request so với bảng in trên console.
