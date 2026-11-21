# Điều chỉnh phong cách của sketch theo nhóm ảnh gallery

Thống kê trung bình và độ lệch chuẩn của patch CLIP ở gallery tạo sáu nhóm phong cách. Một sketch được đọc lại dưới từng nhóm. Hệ thống so phản hồi đó với hiệu ứng cùng phép biến đổi trên các ảnh tham chiếu, rồi dùng nhóm của ảnh ứng viên để xếp hạng.

| mAP / precision (%) | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin 30 lớp |
|---|---:|---:|---:|
| Ghép phong cách theo ảnh ứng viên | 83,00 / 82,23 | 75,01 / 73,56 | 77,91 / 77,66 |
| Dùng một phong cách chung đã tối ưu | 82,95 / 82,11 | 74,85 / 73,44 | 77,81 / 77,57 |
| Khớp từ và diffusion RGB | 81,86 / 81,33 | 74,20 / 72,68 | 79,16 / 80,24 |

Ghép theo ảnh ứng viên tăng mAP lần lượt 0,05, 0,16 và 0,11 điểm so với một phong cách chung. Trên TU-Berlin, phương pháp chính thấp hơn khớp từ + diffusion 1,25 điểm mAP và 2,58 điểm P@100. Hiệu ứng điều kiện hóa theo phong cách còn nhỏ.

Chọn cấu hình trên hai split Sketchy unseen, dùng trung bình mAP rồi đến precision khi hòa. TU-Berlin dùng cấu hình đã khóa. Cả hai benchmark đều đo chiều sketch → photo. OpenAI CLIP ViT-B/32 được đóng băng. Hàm KD-SBIR gốc chấm trên đủ 14.179, 12.694 và 2.130 sketch cho mỗi cấu hình cuối. Ext-1 và TU dùng mAP@all và P@100. Ext-2 dùng mAP@200 và P@200. [Giao thức](../../CANDIDATE_STYLE_PROTOCOL.md) · [Khóa](freeze.json) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU](tuberlin.json) · [Từng truy vấn](per-query.zip)
