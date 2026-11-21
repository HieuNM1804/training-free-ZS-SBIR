# Đọc ảnh ứng viên với cùng trung bình và hiệp phương sai patch như sketch

Thăm dò trên toàn bộ gallery với 900 sketch Ext-1 và 756 sketch Ext-2. Cấu hình của mỗi hàng được chọn theo trung bình mAP của hai mẫu. Từng sketch cung cấp ràng buộc riêng để CLIP đọc lại các ảnh gallery được đề cử.

| mAP / precision (%) | Sketchy Ext-1 | Sketchy Ext-2 |
|---|---:|---:|
| Trừ hiệu ứng xáo bố cục trước và sau chuyển covariance | 84,16 / 83,85 | 80,39 / 77,73 |
| Thay đổi điểm trực tiếp sau chuyển covariance | 84,19 / 83,78 | 80,29 / 77,79 |
| Đọc trực tiếp ảnh được chuyển với từ vựng rút gọn | 84,26 / 83,92 | 80,60 / 77,91 |
| Cùng phép trừ bố cục, chỉ chuyển phương sai từng kênh | 84,04 / 83,73 | 80,47 / 77,57 |
| Cùng phép trừ bố cục, dùng một ảnh tham chiếu gallery cố định | 84,17 / 83,81 | 80,32 / 77,70 |
| Gộp các vùng sketch phóng to trước can thiệp | 84,23 / 83,91 | 80,27 / 77,67 |

So với mốc đầu vào, cơ chế trừ ảnh hưởng bố cục cho thay đổi mAP −0,07 và +0,12 điểm. Cách đọc đơn giản hơn sau khi chuyển covariance cao hơn ở cả hai split. Đây là các số thăm dò. Hiệu quả của cách đọc trực tiếp chưa được đo trên toàn bộ sketch hoặc trên TU. [Giao thức](../../QUERY_OBSERVATION_PROTOCOL.md) · [Cấu hình và kết quả](pilot.json) · [Từng truy vấn](pilot-per-query.zip)
