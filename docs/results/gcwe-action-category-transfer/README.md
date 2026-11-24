# Khi ảnh đúng cùng lớp nhưng khác ảnh mẫu, dùng chung hệ tọa độ có còn giúp truy hồi?

Lượt thăm dò cho thấy dùng chung phân bố ánh xạ cho hai ứng viên cải thiện biên dự báo khi so ảnh mẫu thật với một ảnh khác lớp. Ở đây loại cả 524 ảnh nguồn và đối thủ của lượt thăm dò; mỗi ảnh nguồn mới chỉ đóng góp một sketch. Xếp toàn gallery bằng ngữ nghĩa toàn ảnh và diffusion RGB. Từ top 200, lấy cặp liền kề có một ảnh đúng lớp, một ảnh sai lớp và bỏ cặp chứa ảnh mẫu của sketch. Mỗi sketch đóng góp tối đa một cặp mà xếp hạng ban đầu đúng và một cặp mà nó sai. Các công thức về hình học và suy nghĩa đều khóa trước.

| Kết quả | Ext-1 | Ext-2 |
|---|---:|---:|
| Sketch độc lập / ảnh trong gallery | 2.180 / 17.100 | 1.828 / 12.552 |
| Mốc ngữ nghĩa toàn ảnh, mAP / precision trên toàn gallery | 82,529 / 81,953 | 74,836 / 73,255 |
| Cặp đúng–sai được chọn / số sketch có cặp | 850 / 655 | 998 / 636 |
| Sketch đủ cả hai hướng để tính độ chính xác cân bằng | 195 | 362 |
| Giữ thứ tự toàn ảnh | 50,00% | 50,00% |
| Độ tin cậy của ánh xạ hình học | 49,23% | 50,97% |
| Dự báo nghĩa với tọa độ cố định | 59,74% | 51,24% |
| Dự báo nghĩa khi mỗi ứng viên chọn hệ tọa độ riêng | 57,95% | 46,41% |
| Dự báo nghĩa khi hai ứng viên dùng chung hệ tọa độ | **56,67%** | **46,96%** |

So với hệ tọa độ riêng, dùng chung hệ tọa độ thay đổi −1,28 / +0,55 điểm phần trăm theo sketch. Khi cho các lớp trọng số bằng nhau, khoảng bootstrap 95% của chênh lệch này là [−14,59, 0,95] / [−2,64, 11,09] điểm. Trên tất cả các cặp đã chọn, dùng chung tọa độ sửa 197 và làm sai 223 quyết định ở Ext-1; hai con số ở Ext-2 là 251 và 257. Kết quả thăm dò trên ảnh nguồn không chuyển thành lợi ích theo lớp nhất quán. Theo tiêu chí đã khóa, phép suy luận này chưa đủ để thay đổi bảng xếp hạng toàn gallery hay đánh giá trên TU-Berlin. Trong bảng, chỉ mốc ngữ nghĩa có mAP@all/P@100 ở Ext-1 và mAP@200/P@200 ở Ext-2; các hàng sau là chẩn đoán trên cặp sát hạng.

[Protocol và tiêu chí xác nhận](protocol.md) · [Lượt thăm dò trên ảnh nguồn](../gcwe-shared-intervention-frame/README.md) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [ID và kết quả từng cặp](paired-T4.zip)
