# Đổi qua lại bố cục patch giữa sketch và ảnh ứng viên

Mỗi sketch được chấm trên gallery đầy đủ. Sáu nhóm ảnh được đề cử trước khi CLIP mã hóa lại hai ảnh theo bố cục của nhau. Có đại diện đúng lớp trong 94,1% danh sách trên Ext-1 và 93,0% trên Ext-2. Mẫu thăm dò có 900 và 756 sketch. Mỗi hàng được chọn theo trung bình hai split trong cùng một nhóm đối chứng.

| mAP / precision (%) | Sketchy Ext-1 | Sketchy Ext-2 |
|---|---:|---:|
| Đối chiếu bốn bố cục và trừ hai hoán vị đối chứng | 84,26 / 83,86 | 80,12 / 77,79 |
| Chỉ đối chiếu bốn bố cục | 84,03 / 83,86 | 80,63 / 77,93 |
| Căn lại sketch một chiều với từ điển rút gọn | 84,16 / 84,37 | 80,36 / 77,89 |
| Chỉ dùng độ giống patch được gán | 78,28 / 78,95 | 70,60 / 67,53 |
| Phương pháp gộp các vùng phóng to trước can thiệp | 84,23 / 83,91 | 80,27 / 77,67 |

Cơ chế ghép hai chiều rồi hiệu chỉnh bằng hoán vị đối chứng thấp hơn phương pháp gốc về mAP trung bình. Đối chứng ghép một chiều và bốn bố cục chưa hiệu chỉnh cho kết quả đáng chú ý trên Ext-2 nhưng hiệu quả giữa hai split khác nhau. Dừng hướng ghép có hiệu chỉnh ở thăm dò. [Giao thức](../../RECIPROCAL_REASSEMBLY.md) · [Kết quả từng cấu hình](pilot.json) · [Kết quả từng truy vấn](pilot-per-query.zip)
