# Lựa chọn từ khi nhiễu của các tên gần nhau có tương quan

Đối chứng dùng softmax thông thường. Biến thể cho hai tên gần nhau theo CLIP nhận nhiễu tương quan, lấy từ thắng trong 256 lượt, rồi dùng likelihood sketch của những từ thắng để chấm ảnh. Trong cả hai phương pháp, phép diffusion giữ nguyên. [Mô hình và đối chứng](../../LEXICAL_CHOICE_PROTOCOL.md)

| mAP / precision (%) | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin 30 lớp |
|---|---:|---:|---:|
| Baseline | 81,86 / 81,33 | 74,19 / 72,68 | 79,16 / 80,24 |
| Chọn từ với nhiễu tương quan | 81,67 / 80,95 | 74,45 / 72,58 | 78,42 / 78,96 |
| Giữ độ tập trung bằng cách đổi nhiệt độ | 81,76 / 81,03 | 74,35 / 72,51 | 78,51 / 79,11 |
| Xáo hình học giữa các tên | 81,78 / 81,05 | 74,13 / 72,35 | 78,68 / 79,28 |
| Prior MLE từ gallery + cùng diffusion | 83,25 / 81,98 | 74,89 / 73,53 | 78,09 / 77,69 |

Khi nhân tám số bản sao ở một phần tám từ chọn ngẫu nhiên, baseline đổi top-1 ở 4.112/14.179, 4.840/12.694 và 890/2.130 sketch. Việc chèn bản sao có cùng hai embedding giữ nguyên thứ hạng theo công thức của biến thể; kiểm tra trực tiếp từ thắng trên mỗi split có 0 thay đổi trong 12.288 lượt. Repo vốn đã loại chuỗi giống hệt nhau. Can thiệp vào 1.024 cặp từ có thật, gần nhau trong cả hai không gian prompt, là phép kiểm tra sát thực tế hơn. Ví dụ có cả cặp *door / home door* và các đời xe khác nhau. Khoảng cách embedding chưa xác nhận chúng đồng nghĩa.

Khi bỏ một tên mỗi cặp, baseline đổi ảnh hạng nhất 729, 1.063 và 348 lần. Biến thể đổi 441, 807 và 265 lần. Chỉ đổi seed trong biến thể cũng làm đổi 1.311, 1.047 và 290 lần. Hình học đúng chưa tạo lợi thế truy hồi ổn định trước các đối chứng. Hướng này cung cấp phép đo độ nhạy với từ điển; thuật toán lấy mẫu còn phụ thuộc seed.

Ext-1 và TU dùng mAP@all / P@100, Ext-2 dùng mAP@200 / P@200. Cố định tham số trước khi chấm hai split Sketchy và ghi khóa trước TU. Tất cả truy vấn được đánh giá bằng hàm KD-SBIR gốc trên Colab L4. [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU](tuberlin.json) · [Khóa](freeze.json) · [Từng truy vấn](per-query.zip)
