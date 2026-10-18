# Hướng dẫn đóng góp

Tài liệu này mô tả quy trình làm việc chung của nhóm: cách chia nhánh, mở Pull Request, viết commit message và quy ước viết code. Cách cài đặt môi trường xem ở mục 5 của [README](README.md).

## 1. Quy trình nhánh và Pull Request

```text
feature branch ──PR──▶ develop ──PR──▶ main
```

- `main`: bản ổn định, được bảo vệ. Chỉ merge qua Pull Request, cần 1 approve và CI xanh.
- `develop`: nhánh tích hợp. Mọi thay đổi đều đi vào đây trước.
- Nhánh làm việc: tạo từ `develop`, mỗi nhánh ứng với một task trong kế hoạch, đặt tên theo dạng `<type>/<mô-tả-ngắn>`, ví dụ `feat/predict-endpoint`, `fix/data-leakage`.

Các bước cho một task:

```bash
git switch develop
git pull
git switch -c feat/predict-endpoint
# viết code, test, commit
git push -u origin feat/predict-endpoint
```

Sau đó mở Pull Request vào `develop`:

- Tiêu đề PR theo cùng quy ước với commit message (mục 2).
- Phần mô tả ghi mã task trong kế hoạch và tóm tắt thay đổi.
- PR được merge khi CI xanh và có ít nhất 1 thành viên approve. Dùng **Squash and merge** để mỗi PR thành một commit trên `develop`.

Trưởng nhóm định kỳ mở PR từ `develop` vào `main`.

## 2. Quy ước commit

Mỗi commit chỉ chứa thay đổi của **một task**. Commit message viết bằng tiếng Anh, gồm một dòng duy nhất:

```text
type(scope): summary
```

- `summary` viết ở thể mệnh lệnh, chữ thường ở đầu, không có dấu chấm cuối câu. Message nên ngắn gọn nhưng phải nêu rõ thay đổi gì.
- `type`:

| Type | Dùng cho |
|---|---|
| `feat` | Tính năng mới |
| `fix` | Sửa lỗi |
| `refactor` | Đổi cấu trúc code, không đổi hành vi |
| `perf` | Cải thiện hiệu năng |
| `test` | Thêm hoặc sửa test |
| `docs` | README, MkDocs, docstring |
| `build` | Dependency, Dockerfile, docker compose |
| `ci` | GitHub Actions |
| `chore` | Bảo trì repo: `.gitignore`, pre-commit, cấu hình công cụ |
| `style` | Chỉ thay đổi định dạng |

- `scope` là thành phần bị tác động: `data`, `model`, `train`, `registry`, `api`, `frontend`, `drift`, `monitoring`, `airflow`, `docker`, `deps`, `docs`. Bỏ `scope` khi thay đổi áp dụng cho cả repo.

Ví dụ:

```text
feat(api): add /health and /predict endpoints with pydantic input validation
fix(data): drop reservation_status columns before the temporal split to prevent leakage
test(train): check that the pipeline outputs probabilities in [0, 1]
chore: ignore AI assistant tooling directories
```

## 3. Quy ước code

- Python 3.12. Dependency được quản lý bằng `uv`: dùng `uv add <package>` (hoặc `uv add --dev <package>` cho công cụ phát triển). Luôn commit `pyproject.toml` cùng với `uv.lock`.
- Lint và format bằng `ruff`; độ dài dòng tối đa 120 ký tự.
- Mọi hàm đều có type hint; CI chạy `mypy` trên toàn repo.
- Mọi hàm và class đều có docstring theo Google style.
- Dùng f-string để định dạng chuỗi.
- Code mới đi kèm test trong `tests/`; chạy bằng `uv run invoke test`.
- Không commit dữ liệu, artifact mô hình hoặc secret. Dữ liệu được quản lý bằng DVC, biến bí mật đặt trong `.env` (đã được ignore). Pre-commit chặn file lớn.

## 4. Pre-commit

Mỗi thành viên cài hook một lần sau khi clone:

```bash
uv run pre-commit install
```

Trước mỗi commit, hook sẽ xóa khoảng trắng thừa, kiểm tra YAML, chặn file lớn, chạy `ruff` (lint, có tự sửa) và `ruff format`. Nếu hook sửa file, hãy `git add` lại các file đó rồi commit lần nữa. Để chạy hook trên toàn bộ repo:

```bash
uv run pre-commit run --all-files
```
