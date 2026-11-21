# ICFRR với cùng đặc trưng và hàm chấm

Giữ ViT-B/32 đóng băng và gallery RGB. Chạy ICFRR theo tham số mặc định trong mã của tác giả, với 250 ảnh truy vấn ban đầu, 256 láng giềng, hệ số 0,5 và 5 lượt. Hai cách đưa điểm khớp từ vào cùng reranker là giữ nguyên thang điểm theo CLIP và chia cho điểm ảnh cao nhất của từng sketch. Cả hai dùng cùng từ điển và không cần nhãn. [Thiết kế và mã công khai](../../ICFRR_WORD_CONTROL.md)

| Cấu hình, mAP / precision (%) | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin 30 lớp |
|---|---:|---:|---:|
| ICFRR từ khoảng cách RGB, 5 lượt | 48,29 / 55,11 | 47,99 / 47,72 | 44,80 / 53,59 |
| ICFRR từ điểm khớp từ, 5 lượt | 78,09 / 80,81 | 70,17 / 69,25 | 71,53 / 75,10 |
| ICFRR từ điểm khớp từ chia theo cực đại của sketch, 5 lượt | 78,17 / 81,10 | 71,52 / 69,97 | 73,15 / 77,57 |
| Khớp từ + diffusion RGB của dự án | 81,86 / 81,33 | 74,19 / 72,68 | 79,16 / 80,24 |
| Xác nhận theo cùng từ | 82,56 / 81,46 | 74,07 / 72,73 | 79,77 / 80,62 |

Trong phép thử mặc định, phần lớn khoảng cách với ICFRR đã có ở baseline khớp từ + diffusion RGB. Để định lượng riêng đóng góp mới cần so hai dòng cuối và các đối chứng ràng buộc cùng từ. Code vector hóa ICFRR cho đúng mọi vị trí trong cả 5 lượt so với hai hàm nguyên bản trên 4 truy vấn thật và 513 ảnh thật của từng split. Các điểm chính được chấm bằng hàm KD-SBIR gốc trên tất cả truy vấn và gallery. Các tệp JSON còn ghi điểm ở lượt 0 và 1, và khoảng tin cậy của chênh lệch theo lớp. Tham số và backbone khác có thể cho số liệu khác; bài ICFRR cũng có kết quả cho backbone huấn luyện của tác giả. [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU](tuberlin.json) · [Khóa](freeze.json) · [Từng truy vấn](per-query.zip)
