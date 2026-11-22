# Thử nghiệm trên mẫu với từ nội suy trong bộ mã hóa văn bản CLIP

| mAP / precision (%) | Sketchy Ext-1 | Sketchy Ext-2 |
|---|---:|---:|
| Đối chứng từ vùng sketch | 84,23 / 83,90 | 80,27 / 77,67 |
| Chọn nội suy theo bằng chứng ảnh giữ lại, có trừ độ phổ biến toàn gallery | 83,96 / 84,14 | 80,18 / 77,65 |
| Chọn nội suy theo mức khớp trên ảnh giữ lại của riêng nhóm | **84,39 / 84,54** | **80,65 / 77,92** |

Mỗi nhóm ảnh không nhãn đề xuất tám tên Open Images. Các tên có cùng số token và cùng biên trong prompt được trộn trước bộ mã hóa văn bản. Ảnh khác trong cùng nhóm đo xem CLIP khớp với cách đọc mới hay với hỗn hợp các từ thông thường. Với tám tên, trong hai lần chia gallery trên mỗi split, 76,7–81,9% khối xác suất thuộc các cụm có nhiều tên. Chỉ 31,1–41,2% cụm nhiều tên tăng chỉ số đã chuẩn hóa theo toàn gallery.

Hai split dùng 900 và 756 truy vấn mẫu, cùng toàn bộ gallery 17.100 và 12.552 ảnh. Các cấu hình được chọn trên chính các mẫu này. Kết quả gợi ý thử đầy đủ cách chấm trên riêng nhóm và so với phép trộn embedding sau bộ mã hóa dưới cùng cơ chế chọn. [Giao thức ban đầu](../../VERIFIED_TEXT_INTERPOLATION.md) · [Thiết kế kiểm chứng](../../LOCAL_TEXT_GATE_FOLLOWUP.md) · [Tất cả cấu hình](pilot.json) · [Dữ liệu từng truy vấn](pilot-per-query.zip)
