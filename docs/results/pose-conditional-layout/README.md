# So sánh bằng chứng bố cục của cùng một sketch qua nhiều tư thế

Với từng sketch, CLIP đọc ảnh ở tám phép xoay/lật và bốn cách sắp xếp patch tương ứng. Mô hình đề xuất kết hợp mức ủng hộ cùng một bố cục qua các tư thế. Cấu hình thắng trên Sketchy dùng ảnh gốc và ảnh lật ngang.

| mAP / precision (%) | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin 30 lớp |
|---|---:|---:|---:|
| Bố cục được hai tư thế cùng ủng hộ | 84,11 / 83,05 | 75,69 / 74,27 | 78,76 / 78,55 |
| Tương phản trực tiếp với bố cục bị xáo | 84,26 / 82,96 | 76,00 / 74,40 | 80,10 / 79,63 |
| Chỉ gộp hai tư thế | 84,13 / 83,08 | 75,76 / 74,32 | 78,85 / 78,63 |
| Khớp từ và diffusion RGB | 81,86 / 81,32 | 74,19 / 72,68 | 79,16 / 80,24 |

So với chỉ gộp hai tư thế với cùng prior, cơ chế đồng thuận làm mAP thay đổi −0,02, −0,07 và −0,09 điểm. Đối chứng tương phản trực tiếp đạt 80,10 mAP trên TU với cùng cấu hình đã khóa trên Sketchy. Mức cải thiện thuộc về cách đọc các ảnh biến đổi. Sự đồng thuận giữa các tư thế chưa cho lợi ích riêng.

Chọn cấu hình trên hai split Sketchy unseen, dùng trung bình mAP rồi đến precision khi hòa. TU-Berlin dùng cấu hình đã khóa. Cả hai benchmark đều đo chiều sketch → photo. OpenAI CLIP ViT-B/32 được đóng băng. Hàm KD-SBIR gốc chấm trên đủ 14.179, 12.694 và 2.130 sketch cho mỗi cấu hình cuối. Ext-1 và TU dùng mAP@all và P@100. Ext-2 dùng mAP@200 và P@200. [Giao thức](../../POSE_LAYOUT_PROTOCOL.md) · [Khóa](freeze.json) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU](tuberlin.json) · [Từng truy vấn](per-query.zip)
