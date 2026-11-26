# Kết quả phép thay thế cặp quan sát trong gallery

Cơ chế thay lần đọc nguồn của một ảnh gallery bằng sketch truy vấn đạt kết quả dưới Dot và dưới ánh xạ cố định trên cả ba split full unseen. Bỏ chính ảnh ứng viên khỏi ma trận căn chỉnh thay đổi mAP ít hơn 0,04 điểm trên mỗi split. Kết quả bác bỏ việc dùng chi phí thay thế này làm tín hiệu chính của truy hồi theo lớp. [Công thức và giả thuyết](../../gallery-substitution-alignment.md) · [Prior art và phản ví dụ](../../gallery-substitution-prior.md)

| Cấu hình | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin 30 lớp |
|---|---:|---:|---:|
| Dot 601 từ | 75,35 / 77,10 | 72,66 / 71,08 | 73,46 / 73,98 |
| Thay lần đọc của ứng viên | 54,98 / 64,29 | 57,14 / 53,26 | 55,05 / 61,71 |
| Không trừ mức khớp của chính ảnh | 56,07 / 65,97 | 60,00 / 53,68 | 57,48 / 66,05 |
| Ánh xạ cố định giữa hai bộ đọc | 66,30 / 71,73 | 64,80 / 62,22 | 67,69 / 71,20 |
| Ánh xạ cố định có trừ mốc tự nhiên | 64,79 / 71,03 | 63,42 / 60,88 | 65,58 / 69,44 |
| Ứng viên vẫn nằm trong ma trận | 55,01 / 64,33 | 57,18 / 53,29 | 55,08 / 61,74 |

Ext-1 và TU dùng mAP@all / P@100. Ext-2 dùng mAP@200 / P@200. Chấm đủ 14.179 / 12.694 / 2.130 sketch trên 17.100 / 12.552 / 27.262 ảnh. Colab T4, seed 42, cả hai bộ cùng đo sketch sang ảnh. Các kết quả riêng từng truy vấn và SHA được lưu trong mỗi thư mục. [Lượt chính](freeze.json) · [Đối chứng](attribution/) · [Chi tiết TU-Berlin](tuberlin.json)

## Kiểm tra trên chính các ảnh đã biết đối sánh

Dùng ảnh RGB đọc qua prompt nguồn làm truy vấn và tìm đúng ảnh đó qua prompt đích. Theo công thức, điểm của chính nó bằng 0. Một ảnh khác vượt cả 0 lẫn điểm tự khớp với biên lớn hơn 0,0002 trong 16.422 / 17.100 trường hợp Ext-1, 11.634 / 12.552 ở Ext-2 và 26.776 / 27.262 ở TU-Berlin. Sai lệch tuyệt đối so với SVD trực tiếp ở ví dụ mạnh nhất từng split đều dưới 0,00002. Điểm dương phản ánh mức thay đổi độ khớp của gallery khi thay lần đọc. Nó có thể ưu tiên cả một ứng viên khác lớp hơn chính ảnh nguồn. [Kiểm tra tự khớp](native-match-audit.json)

Commit chạy lượt chính là 499c7a1, đối chứng là 65cf997 và kiểm tra tự khớp là 7c37e02. Nhãn chỉ dùng sau khi có thứ hạng.
