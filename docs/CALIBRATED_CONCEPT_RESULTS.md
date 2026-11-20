# So khớp sketch và ảnh bằng phân bố trên từ vựng CLIP

Hai ảnh thuộc hai miền có thể gần nhau nếu cùng đặt xác suất cao lên một số khái niệm. Nhánh này kiểm tra xem cách so phân bố có thu hồi được ảnh đúng và việc giảm trọng số của những từ phổ biến trong gallery có giúp hay không. Tiền lệ là biểu diễn văn bản và ảnh qua các lớp ngữ nghĩa cho cross-modal retrieval, chuyển nội dung ảnh qua từ vựng rời rạc và diffusion trên gallery. Ở đây, encoder CLIP ViT-B/32 đóng băng cung cấp toàn bộ embedding. [Rasiwasia và cộng sự, ACM MM 2010](https://www.mit.edu/~rplevy/papers/rasiwasia-etal-2010-acm.pdf) · [FreeDom, WACV 2025](https://arxiv.org/abs/2412.03297) · [Iscen và cộng sự, CVPR 2017](https://arxiv.org/abs/1611.05113)

Với mỗi sketch, CLIP đọc các từ theo prompt sketch. Mỗi ảnh gallery có một phân bố riêng trên cùng từ theo prompt photo. Điểm chung là tổng các tích giữa xác suất từ của sketch và xác suất từ của ảnh, có thể chia cho một lũy thừa của tần suất từ trong gallery. Điểm log sau chuẩn hóa được phối hợp với trung bình của ba phép so trực tiếp gồm chuyển sketch sang prompt ảnh, chuyển ảnh sang prompt sketch và so hai mã từ vựng trong miền ảnh. Cuối cùng, nhánh này cho lan truyền điểm của từng truy vấn trên đồ thị ảnh.

| Cấu hình khóa | Sketchy Ext-1 P@100 / mAP@all | Sketchy Ext-2 P@200 / mAP@200 | TU-Berlin P@100 / mAP@all |
|---|---:|---:|---:|
| Phân bố khái niệm và diffusion | 75,96 / 76,37 | 67,89 / 69,16 | 76,77 / 75,62 |

Phép chọn một cấu hình chung theo trung bình precision trên hai split Sketchy, xét mAP nếu hòa, trả về số mũ hiệu chỉnh tần suất bằng 0. Kết quả ủng hộ việc khai thác từ chung. Chưa có bằng chứng về lợi ích từ phép chia tần suất. Bỏ diffusion đưa precision về 74,89 / 66,41 / 75,41. Xáo quan hệ giữa hai bộ prompt đưa precision về 2,65 / 3,13 / 2,86.

Đối chứng dùng ảnh gallery để chọn từ rồi biểu diễn ảnh trong miền sketch dẫn đến nhánh nghiên cứu tiếp theo. Các tham số của cả hai nhánh được khóa trên Sketchy trước khi chấm TU-Berlin. [Kết quả và đối chứng](results/calibrated-concept-retrieval/concept-results.json) · [Cấu hình khóa](results/calibrated-concept-retrieval/freeze.json) · [Mã thực nghiệm](../tf_sbir/concept_likelihood.py)
