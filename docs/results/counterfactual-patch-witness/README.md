# Đổi bộ đọc attention của CLIP giữa sketch và ảnh để kiểm tra bằng chứng patch

Cho hai bộ đọc CLS gốc của ViT-B/32 quan sát cả patch sketch và patch ứng viên. So sánh sự thay đổi hướng đọc trên hai ảnh, có hiệu chỉnh bằng một bảng ảnh gallery tham chiếu. Cấu hình đã khóa can thiệp vào 512 ảnh hạng đầu rồi lan truyền lại thay đổi tới toàn gallery.

| mAP / precision (%) | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin 30 lớp |
|---|---:|---:|---:|
| Khớp từ và diffusion gốc | 81,86 / 81,33 | 74,19 / 72,68 | 79,16 / 80,24 |
| Đổi bộ đọc, có hiệu chỉnh tham chiếu | 82,02 / 81,68 | 74,29 / 72,72 | 79,47 / 80,61 |
| Đổi bộ đọc, bỏ hiệu chỉnh tham chiếu | 82,15 / 81,84 | 74,47 / 72,96 | 79,47 / 80,44 |
| Một bộ đọc chung cho mọi ảnh | 82,08 / 81,69 | 74,44 / 72,87 | 79,48 / 80,83 |
| Bộ đọc gốc của từng ảnh | 82,03 / 81,56 | 74,32 / 72,70 | 79,32 / 80,63 |
| Đọc bằng một ảnh khác | 81,91 / 81,44 | 74,16 / 72,57 | 79,40 / 80,84 |

Mức tăng mAP của can thiệp chính so với baseline lần lượt là 0,16, 0,09 và 0,30 điểm. Đối chứng một bộ đọc chung cũng tăng và cho mAP cao hơn ở cả ba split. Hiệu chỉnh theo bảng tham chiếu chưa chứng minh giá trị riêng. Các bảng JSON ghi tỷ lệ AP từng sketch tăng/giảm và so sánh ghép cặp với tất cả đối chứng.

CLIP ViT-B/32 đóng băng. Mỗi cấu hình cuối được chấm bằng hàm KD-SBIR gốc trên 14.179, 12.694 và 2.130 sketch ở Colab L4. Ext-1 và TU dùng mAP@all và P@100. Ext-2 dùng mAP@200 và P@200. Chọn trên hai split Sketchy unseen rồi khóa cấu hình trước khi chấm TU. Ở cả hai benchmark, chiều truy hồi là sketch → photo. [Giao thức](../../COUNTERFACTUAL_PATCH_PROTOCOL.md) · [Khóa](freeze.json) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU](tuberlin.json) · [Từng truy vấn](per-query.zip)
