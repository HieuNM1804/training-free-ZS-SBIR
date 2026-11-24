# Suy ra nghĩa của sketch từ bố cục các patch CLIP

Lấy một patch làm tâm, xếp 48 patch còn lại theo mức giống nhau trong CLIP. Với từng cặp hạng đối xứng, ghi hai khoảng cách trên lưới từ tâm đến patch giống hơn và patch khác hơn. Mỗi khái niệm công khai dự báo một phân bố của hai khoảng cách dựa trên ảnh của một nửa gallery. Sketch thay đổi xác suất của một khái niệm khi bố cục quan sát được phù hợp với dự báo cho khái niệm đó. Hậu nghiệm dùng để truy hồi ở nửa gallery còn lại. Cấu hình chính tích phân trên một sự kiện hình học. Cấu hình phụ dùng số quan sát hiệu dụng 16 và đã được khóa trước khi chấm xác nhận cùng TU-Berlin.

| Truy hồi trực tiếp, mAP / precision (%) | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin 30 lớp |
|---|---:|---:|---:|
| Khớp từ của CLIP | 75,31 / 78,94 | 71,41 / 70,28 | 73,35 / 77,22 |
| GCWE, một quan sát (chính) | 75,46 / 79,14 | 71,61 / 70,43 | 73,38 / 77,30 |
| GCWE, 16 quan sát hiệu dụng | 76,28 / 80,34 | 72,47 / 71,15 | 73,29 / 77,66 |
| Cùng 16 quan sát, coi hai khoảng cách độc lập | 76,24 / 80,25 | 72,46 / 71,14 | 73,33 / 77,67 |

Kết quả dùng mọi truy vấn chính thức, lần lượt 14.179 / 12.694 / 2.130 sketch với 17.100 / 12.552 / 27.262 ảnh. Hai split Sketchy đã tham gia phát triển. Ở 16 quan sát, so với khớp từ trực tiếp mAP đổi +0,97 / +1,06 / −0,06 điểm. Khoảng bootstrap 95% khi cho trọng số bằng nhau theo lớp là [+0,65, +1,33], [+0,06, +1,86] và [−0,91, +0,64]. Phần tăng thêm khi mô hình hóa đồng thời hai khoảng cách rất nhỏ. So với hai phân bố độc lập, mAP đổi +0,039 / +0,015 / −0,043 điểm.

| Cùng đồ thị RGB, mAP / precision (%) | Ext-1 xác nhận | Ext-2 xác nhận | TU 30 lớp |
|---|---:|---:|---:|
| Khớp từ và diffusion | 81,86 / 81,37 | 74,27 / 72,73 | 79,16 / 80,23 |
| Thêm quan sát hình học đơn vào hậu nghiệm từ | 81,93 / 81,47 | 74,32 / 72,81 | 79,14 / 80,20 |
| Dùng 16 quan sát hiệu dụng | 82,33 / 81,83 | 74,36 / 72,86 | 78,77 / 79,76 |
| Hoán đổi bố cục giữa các ảnh tham chiếu, vẫn dùng 16 quan sát | 82,24 / 81,73 | 74,16 / 72,69 | 78,75 / 79,83 |

Bảng đồ thị dùng 12.643 / 11.158 sketch còn lại sau pilot và toàn bộ TU. Ở Ext-1, phần lớn lợi ích so với khớp từ vẫn xuất hiện khi gán bố cục cho sai ảnh. Ở Ext-2, chênh lệch đúng trừ hoán đổi lớn hơn mức tăng so với diffusion. Dấu và độ lớn của các đối chứng khác nhau theo split. Dữ liệu chưa chứng minh quan hệ hai khoảng cách là nguyên nhân chính cải thiện truy hồi xuyên miền.

**Dự báo nghĩa của ảnh đúng lớp.** Với 16 quan sát, mức tăng log xác suất cho một từ của ảnh đúng lớp là 0,261 / 0,365 / 0,145 nat. Nhưng cập nhật cũng giúp dự báo từ của ảnh khác lớp. Phần tăng thêm cho ảnh đúng so với ảnh khác lớp chỉ là 0,004 / 0,011 / 0,018 nat. Khoảng 95% theo lớp của cả ba chênh lệch đều chứa 0. Điểm tốt hơn trên Sketchy chưa đủ xác nhận rằng hình học chỉ ra khái niệm đặc trưng của đối tượng.

Các lượt chạy dùng Colab T4, CLIP ViT-B/32 đóng băng, Open Images 20.931 từ và seed 42. Ext-1 và TU dùng mAP@all / P@100; Ext-2 dùng mAP@200 / P@200. Mã pilot `5e7c8ac`, xác nhận và chuyển giao `1718a22`, kiểm tra toàn bộ ID và đích ngữ nghĩa `32af871`. [Nguyên lý và prior art](protocol.md) · [Lựa chọn trước khi xem xác nhận](selection.md) · [Đích dự báo ngữ nghĩa](semantic-endpoint.md) · [Ext-1 đầy đủ](sketchy_1-full.json) · [Ext-2 đầy đủ](sketchy_2-full.json) · [TU đầy đủ](tuberlin-full.json) · [Metric và log score theo từng truy vấn](paired-T4.zip)
