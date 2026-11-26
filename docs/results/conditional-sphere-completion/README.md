# Tái dựng embedding ảnh gallery có điều kiện

Đây là thí nghiệm từng được báo tăng **4,30 điểm mAP trên Sketchy Ext-1**. Cùng cấu hình tăng 3,08 điểm trên Ext-2 và 4,01 điểm trên TU-Berlin đủ 30 lớp. Trung bình với trọng số bằng nhau của hai mức tăng trên Sketchy là 3,69 điểm. Mỗi ô dưới đây ghi mAP / precision (%).

| Cấu hình | Sketchy Ext-1 · all / 100 | Sketchy Ext-2 · 200 / 200 | TU-Berlin · all / 100 |
|---|---:|---:|---:|
| Semantic Dot 601 từ | 75,35 / 77,10 | 72,66 / 71,08 | 73,46 / 73,98 |
| Tái dựng từ các ảnh lân cận | **79,65 / 80,07** | **75,74 / 73,61** | **77,46 / 76,47** |
| Trung bình phân bố từ của đúng các ảnh lân cận ấy | 79,87 / 80,16 | 75,80 / 73,65 | 77,67 / 76,57 |

Với mỗi ảnh gallery, phép chiếu trực giao giữ một không gian con 64 chiều của embedding CLIP. Trong không gian này, chọn 64 ảnh lân cận và lần lượt mượn hướng của phần embedding nằm ngoài không gian con từ từng ảnh. Độ dài phần mượn bằng độ dài phần ẩn của ảnh ban đầu. CLIP đọc mỗi embedding tái dựng thành phân bố trên 601 danh từ Open Images. Lấy trung bình các lần đọc trên 64 ảnh và bốn phép chiếu rồi chấm bằng tích vô hướng với phân bố từ của sketch. Hai template là `a drawing of a {}.` và `a photo of the {}.`.

Đối chứng giữ nguyên bốn phép chiếu, toàn bộ danh tính ảnh lân cận, từ điển, sketch và evaluator. Chỉ thay phân bố từ của embedding tái dựng bằng phân bố từ của ảnh lân cận gốc. Đối chứng cao hơn phép tái dựng 0,22 / 0,06 / 0,21 điểm mAP. Dữ liệu ủng hộ lợi ích của thông tin từ lân cận gallery. Bằng chứng hiện có chưa xác lập đóng góp riêng của phần tái dựng.

Chạy trên Colab T4, OpenAI CLIP ViT-B/32 đóng băng, seed 42. Chọn số chiều và số lân cận bằng trung bình mAP trên toàn bộ hai split Sketchy unseen, khóa trước khi chấm TU-Berlin. Ext-1 có 14.179 sketch / 17.100 ảnh, Ext-2 có 12.694 / 12.552, TU-Berlin có 2.130 / 27.262. Các số và đối chứng trong bảng được trích từ log của lượt chạy cùng thiết bị. File JSON giữ độ chính xác ba chữ số thập phân đã ghi trong log. [Kết quả từng cấu hình](full-unseen.json) · [Mã phương pháp](../../../tf_sbir/conditional_completion.py) · [Giao thức chọn và chấm](../../../tf_sbir/conditional_completion_study.py)
