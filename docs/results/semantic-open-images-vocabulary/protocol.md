# So sánh kích thước từ điển trong S

Ba cấu hình dùng nguyên CLIP ViT-B/32 đóng băng và hai bộ prompt miền của S. Cấu hình đầy đủ dùng tất cả 20.931 tên duy nhất. Cấu hình vật thể dùng toàn bộ 601 tên trong CSV boxable. Cấu hình ngẫu nhiên lấy 601 chỉ số không hoàn lại từ tập đầy đủ với một `torch.Generator` CPU seed 42. Hai danh sách thu gọn được lưu trước khi chấm. Danh sách dùng chung cho mọi split và chỉ dựa trên hai CSV công khai. [Open Images V7](https://storage.googleapis.com/openimages/web/download_v7.html)

Mỗi sketch sinh hai phân bố từ trên ảnh gốc và ảnh lật ngang rồi lấy trung bình. Photo dùng một phân bố trên ảnh gốc. Với mỗi cấu hình, softmax áp dụng trên đúng tập từ của cấu hình đó trong cả hai miền. Lấy tích vô hướng các xác suất để xếp hạng toàn bộ gallery cho từng sketch. Nhiệt độ, prompt, xử lý ảnh và feature CLIP giống nhau giữa ba cấu hình.

Chấm full unseen bằng KD-SBIR gốc trên Sketchy Ext-1 (14.179 sketch, 17.100 photo), Ext-2 (12.694 sketch, 12.552 photo) và TU-Berlin Ext (2.130 sketch, 27.262 photo, 30 lớp). Ext-1 và TU-Berlin dùng mAP@all và P@100. Ext-2 dùng mAP@200 và P@200. So sánh chênh lệch có ghép cặp từng sketch và bootstrap AP trung bình theo lớp. Giữ nguyên ba cấu hình cho cả ba split.

Chạy trên Colab với dữ liệu và cache của repo tại `/content/sbir`. Dùng các file `assets/open-images-v7-classes.csv` và `assets/open-images-v7-boxable.csv` lấy từ Open Images V7.

```sh
export PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1
python -u -m tf_sbir.semantic_open_images_vocabulary sketchy
python -u -m tf_sbir.semantic_open_images_vocabulary tuberlin
```
