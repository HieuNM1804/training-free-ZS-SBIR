# Loại ảnh bằng những phân biệt ngữ nghĩa mà nó không thể truyền

Đặt một câu hỏi nhị phân cho mỗi cặp từ trong 64 từ của gallery. Nếu chỉ biết vị trí ở một trong bốn vùng, xác suất đoán đúng từ nào đã xuất hiện là (1 + TV)/2 khi hai từ có xác suất bằng nhau. Theo bất đẳng thức xử lý dữ liệu, việc làm mất thông tin bên ảnh không thể khiến quyết định dễ hơn bên sketch. Với mỗi ứng viên, lấy vi phạm lớn nhất trong 2.016 quyết định làm chặn dưới cho sai số mô phỏng sketch. Điều kiện này dùng được ngay cả khi hai miền không chung khung tọa độ. Mỗi truy vấn là một sketch. Các điểm bằng nhau được phân xử bằng thứ hạng ngữ nghĩa toàn ảnh gốc.

| 768 sketch mỗi split trên gallery đầy đủ | Ext-1 mAP@all / P@100 | Ext-2 mAP@200 / P@200 |
|---|---:|---:|
| Ngữ nghĩa toàn ảnh sau diffusion | 80,563 / 80,272 | 72,862 / 71,531 |
| Xếp theo chặn dưới với đúng danh tính từ | 6,684 / 9,717 | 11,971 / 9,126 |
| Tráo tên từ theo từng cặp từ sát hạng trong gallery | 4,335 / 4,419 | 7,356 / 5,218 |
| Ghép các độ phân biệt theo thứ tự cường độ, bỏ danh tính cặp | 47,920 / 74,740 | 72,720 / 59,963 |
| Đặt nhóm không vi phạm trước, giữ thứ tự toàn ảnh trong mỗi nhóm | 80,442 / 80,121 | 72,849 / 71,528 |

Danh tính từ thật cải thiện mAP 2,35 / 4,62 điểm so với tráo tên, nhưng chỉ **0,142% / 0,651%** cặp sketch–ảnh vượt qua tất cả điều kiện. Trong top 200 của ngữ nghĩa toàn ảnh, tỷ lệ tương ứng là 0,142% / 0,651%. Trên các cặp sát hạng có đúng một ảnh đúng lớp, chặn dưới trung bình của ảnh đúng là 0,270 / 0,254; của ảnh sai là 0,266 / 0,260. Chưa thể dùng tính khả thi của mô hình suy giảm thông tin để nhận diện ảnh đúng. Điểm của đối chứng bỏ danh tính có thể hưởng lợi từ thứ hạng toàn ảnh ở các trường hợp hòa, nên không được diễn giải thành hiệu năng riêng của thông tin không gian.

Một giả thuyết tiếp theo bắt nguồn từ cách tạo quan sát. Dù một từ gần như vắng mặt ở mọi vùng, chuẩn hóa riêng từ đó vẫn ép tổng xác suất vị trí bằng một. Mô hình đã kiểm tra không phân biệt được *không thấy từ* và *thấy từ nhưng chưa rõ ở đâu*. [Thiết kế, suy dẫn và nguồn nghiên cứu](protocol.md) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [Kết quả theo từng sketch](paired-T4.zip)
