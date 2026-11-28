# Tài liệu

Nguồn của trang tài liệu công khai **https://quandao073.github.io/IT5414-Group7-MLOps/**.
Cấu hình MkDocs nằm ở `mkdocs.yml` tại gốc repo, nội dung ở `docs/source/`.

Xem local:

```bash
uv run invoke serve-docs     # http://127.0.0.1:8000
```

Build kiểm tra (giống CI, cảnh báo link gãy là lỗi):

```bash
uv run invoke build-docs     # mkdocs build --strict
```

Workflow `.github/workflows/docs.yaml` build và deploy lên GitHub Pages mỗi khi `main` thay
đổi. Bật Pages một lần tại **Settings → Pages → Source: GitHub Actions**.

Thêm trang mới: tạo file trong `docs/source/` rồi khai báo trong mục `nav` của `mkdocs.yml`.
Liên kết tới file ngoài `docs/source/` (README, `reports/`) phải dùng URL GitHub tuyệt đối,
nếu không `--strict` sẽ báo lỗi.
