# Thăm dò bổ sung thông tin bên trong patch

Chấm 900 sketch trên Ext-1 và 756 sketch trên Ext-2, chọn đều 36 truy vấn ở mỗi lớp. Mỗi truy vấn tìm trong toàn bộ gallery. Các cấu hình dưới đây được chọn trong từng nhóm theo trung bình mAP trên hai mẫu. Đây là số thăm dò trên Sketchy.

| mAP / precision (%) | Sketchy Ext-1 | Sketchy Ext-2 |
|---|---:|---:|
| Chi tiết ở lớp 12 với khối lượng attention của patch mẹ cố định | 83,47 / 82,90 | 78,78 / 76,59 |
| Cùng lớp, cho attention chọn tự do giữa mọi patch con | 83,50 / 82,92 | 78,83 / 76,60 |
| CLIP ViT-B/32 ở kích thước gốc, cùng chỉ mục và tỷ lệ trộn | 83,42 / 82,88 | 78,79 / 76,62 |
| Nội suy vị trí và chạy CLIP ở 448 pixel | 83,00 / 82,49 | 78,80 / 76,61 |
| Gộp bốn góc ảnh | 82,79 / 82,65 | 78,17 / 76,11 |

Can thiệp giữ khối lượng attention cho mỗi patch mẹ cho chênh lệch mAP +0,05 và −0,01 điểm so với ảnh gốc. Đối chứng nới ràng buộc cao hơn trên cả hai mẫu. Tôi dừng nhánh này ở thăm dò Sketchy và dành lượt đánh giá đầy đủ cho các hướng cải thiện ổn định hơn. [Cơ chế](../../ATTENTION_DETAIL_PROTOCOL.md) · [Kết quả từng cấu hình](pilot.json) · [Ext-1](pilot-sketchy_1.json) · [Ext-2](pilot-sketchy_2.json)
