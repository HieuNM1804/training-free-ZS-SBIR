# Khi hai hình cùng gợi một từ, chúng có dựa vào cùng attention head?

Mỗi sketch đề xuất tên công khai. Với mỗi tên, chúng tôi tách đóng góp của 48 attention head ở bốn block cuối của CLIP ViT-B/32, sau đó trừ phần dự đoán được từ tổng điểm của tên đó. Dùng ảnh gallery để ước lượng phần dư cho hai prompt và xem một sketch có khớp với ảnh ở chính các head đó hay không. Các tên, thống kê gallery và hệ số suy luận được tính độc lập với nhãn truy hồi. [Nguồn phân rã CLIP, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5085a2b8c298edeadc46b9ffe6df5f64-Abstract-Conference.html) · [Giao thức và nguồn của retriever nền](../../WORD_ATTRIBUTION_AGREEMENT.md)

| mAP / precision (%) | Sketchy Ext-1 | Sketchy Ext-2 |
|---|---:|---:|
| Điểm nền theo vùng | 84,64 / 83,96 | 80,63 / 78,38 |
| Cộng độ khớp phần dư của cùng head | 84,54 / 83,77 | 80,49 / 78,34 |
| Hoán đổi head trong cùng block | 84,51 / 83,88 | 80,20 / 78,38 |
| Cộng điểm tên đầy đủ | 84,61 / 83,98 | 80,58 / 78,34 |
| Gộp head trong từng block, rồi so phần dư | 84,50 / 83,82 | 80,42 / 78,24 |

Hai mẫu cân bằng dùng 36 sketch mỗi lớp, tương ứng 900 và 756 sketch, với gallery đầy đủ. Cấu hình phần dư tốt nhất theo trung bình mAP sử dụng 64 tên, nhiệt độ tương đối 0,3 và mức cộng 0,035. Điểm nền cao hơn trên cả hai mẫu. [Các cấu hình](pilot.json) · [Kết quả từng sketch](pilot-per-query.zip)

Với mỗi sketch, lấy tối đa 32 ảnh đúng phân bố dọc 2.048 vị trí đầu. Ghép từng ảnh đúng với một ảnh sai có điểm nền gần nhất. Phần dư xếp ảnh đúng cao hơn trong 44,02% của 27.583 cặp trên Ext-1 và 45,46% của 23.708 cặp trên Ext-2. Trong khi đó, trung vị tương quan giữa các đáp ứng của hai prompt khi đo trên cùng ảnh gallery đạt khoảng 0,86 ở cả hai split. Thông tin có thể ổn định với một ảnh nhưng dự đoán kém khi hai hình khác miền chỉ cần cùng lớp. Các cặp được lấy từ thứ hạng nền, nên kết quả mô tả điều kiện chọn mẫu này. Nhánh kết thúc ở phép thử Sketchy. [Ext-1](pilot-sketchy_1.json) · [Ext-2](pilot-sketchy_2.json)
