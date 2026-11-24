# Phần nào của lân cận ảnh thật sự ủng hộ một sketch?

Trong Cross-Domain Semantic Retrieval, một sketch cho phân bố từ và mỗi ảnh cho xác suất của việc quan sát các từ đó. Mô hình coi một phần chưa biết của 16 ảnh gần nhau trong gallery là các ảnh mang cùng bằng chứng ngữ nghĩa. Nó tích phân trên tỷ lệ ảnh ủng hộ, rồi xếp hạng toàn gallery cho từng sketch. Cấu hình được chọn thăm dò trên 1.536 truy vấn ở mỗi split Sketchy, sau đó khóa trước khi chấm 12.643 / 11.158 truy vấn còn lại và toàn bộ 30 lớp TU-Berlin. Phép ghép hai phần Sketchy kiểm tra từng ID xuất hiện đúng một lần.

| Hệ thống độc lập, mAP / precision (%) | Ext-1, @all / @100 | Ext-2, @200 / @200 | TU 30 lớp, @all / @100 |
|---|---:|---:|---:|
| Khớp từ trực tiếp | 75,31 / 78,94 | 71,41 / 70,28 | 73,35 / 77,22 |
| Trung bình trên cùng 16 ảnh | 79,87 / 80,64 | 73,30 / 72,10 | 77,77 / 78,47 |
| Tỷ lệ cố định 1/2 | 80,03 / 81,13 | 73,77 / 72,60 | 77,93 / 79,07 |
| Suy luận trên tỷ lệ chưa biết | 79,99 / 81,16 | 73,82 / 72,63 | 77,87 / 79,10 |
| Diffusion RGB đã khóa | **81,86 / 81,33** | **74,21 / 72,68** | **79,16 / 80,23** |

Mức tăng mAP so với khớp từ trực tiếp là 4,68 / 2,40 / 4,52 điểm. Đối chứng ghép cặp cho thấy phần lớn hiệu ứng đến từ việc dùng các ảnh lân cận. So với trung bình trên cùng 16 ảnh, mức tăng còn 0,11 / 0,51 / 0,10 điểm. Khoảng bootstrap 95% theo lớp lần lượt là [−0,12, +0,32], [+0,22, +0,81] và [−0,31, +0,45]. Độ chính xác tăng so với phép lấy trung bình trên cả ba bộ, nhưng mô hình chưa cải thiện đồng thời mAP và precision so với diffusion. Cấu hình cố định tỷ lệ ở 1/2 cho kết quả rất gần. Dữ liệu chưa ủng hộ việc lấy suy luận tỷ lệ chưa biết làm đóng góp phương pháp chính.

Đã chấm 14.179 / 12.694 / 2.130 truy vấn với 17.100 / 12.552 / 27.262 ảnh, dùng CLIP ViT-B/32 đóng băng, từ điển Open Images, seed 42, evaluator KD-SBIR trên Colab T4. [Cơ chế và bài báo liên quan](protocol.md) · [Cấu hình đã khóa](selection.md) · [Hai split Sketchy đầy đủ](complete-sketchy.json) · [TU-Berlin](tuberlin-transfer.json) · [Bản ghi theo truy vấn](paired-T4.zip)
