# Một vùng của ảnh có dự báo được nghĩa của phần sketch bị che?

Cho CLIP xem một dải sketch và dải bổ sung của một ảnh ứng viên. Đích dự báo là phân bố 20.931 từ khi cho CLIP đọc riêng dải sketch còn lại. Chấm bằng log loss kỳ vọng trên toàn bộ từ vựng. Phép dự báo không thấy vùng đích. Đổi chiều và đổi trục để mỗi bên có lượt cung cấp bằng chứng như nhau. Đối chứng chính giữ toàn bộ patch nhưng buộc hai phía chỉ giao tiếp thông qua token tổng hợp CLS. Đối chứng vị trí hoán đổi hai dải của cùng ảnh trước bước gán tọa độ, giữ nguyên các pixel mà ứng viên cung cấp qua hai chiều.

| Cách dự báo phần sketch bị che | Ext-1, đúng cân bằng (%) | Ext-2, đúng cân bằng (%) |
|---|---:|---:|
| Sketch và vùng ảnh, cho các patch tương tác trực tiếp | 51,28 | 48,90 |
| Cùng đầu vào, chỉ giao tiếp gián tiếp qua CLS | 48,46 | 49,72 |
| Đổi vị trí hai dải ảnh, cho các patch tương tác trực tiếp | 48,72 | 46,27 |
| Đổi vị trí, chỉ giao tiếp qua CLS | 43,59 | 47,65 |
| Chỉ dùng dải ảnh ứng viên | 49,49 | 47,65 |
| Chỉ dùng dải sketch đã thấy, hai ứng viên hòa | 50,00 | 50,00 |

Trên 195 / 362 sketch có đủ hai chiều của quyết định tham chiếu, mở tương tác trực tiếp làm **tăng** log loss của ảnh đúng thêm 0,0405 / 0,0215 nat. Mở đường truyền này làm thay đổi lợi thế dự báo của ảnh đúng so với ảnh sai +0,0090 / −0,0172 nat theo truy vấn. Khoảng bootstrap 95% theo lớp là [−0,0680, +0,0523] / [−0,0993, +0,0401]. So với đổi vị trí dải ảnh, lợi thế đúng trừ sai là +0,0309 / +0,0270 nat theo truy vấn, nhưng hai khoảng theo lớp đều chứa 0. 850 / 998 cặp so sánh sát hạng được lưu trong bản ghi.

Khi thêm vùng ảnh đúng vào sketch, log loss còn kém cả dự báo chỉ từ vùng sketch đã thấy 1,1369 / 1,2266 nat. Ngược lại, khi lấy ảnh đơn lẻ làm mốc rồi thêm sketch, log loss giảm 0,5464 / 0,6325 nat. Điều này cho thấy vì sao chỉ so với đầu vào ảnh có thể dẫn đến kết luận sai về thông tin do ứng viên cung cấp. Bộ mã hóa còn có thể biểu diễn thông tin mà bộ đọc từ trực tiếp không khai thác được. Hai split của tập cặp này chưa cho bằng chứng để đưa cơ chế sang TU-Berlin.

I-JEPA đã dùng ngữ cảnh để dự báo biểu diễn một vùng bị che khi huấn luyện. Phép thử ở đây cố định toàn bộ CLIP, cho ứng viên khác miền cung cấp một phần ngữ cảnh, rồi kiểm tra giá trị tăng thêm của ứng viên bằng câu trả lời trên chính vùng sketch chưa thấy. Đây là một đối chứng về nguồn bằng chứng. Chưa xác lập một thuật toán truy hồi mới có khả năng chuyển giao. [I-JEPA, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.pdf)

Mã `88f045e`, Colab Tesla T4, frozen OpenAI CLIP ViT-B/32, seed 42. [Protocol đã khóa](protocol.md) · [Kết quả Ext-1](sketchy_1.json) · [Kết quả Ext-2](sketchy_2.json) · [Bản ghi đủ bốn chiều theo từng ứng viên](paired-T4.zip)
