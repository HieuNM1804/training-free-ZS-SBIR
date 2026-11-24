# Dự báo crop có giải quyết nhầm lớp trong truy hồi không?

Dùng 768 truy vấn mỗi split, hai gallery đầy đủ gồm 17.100 / 12.552 ảnh. Lập các cặp hạng (1,2), (3,4), …, (199,200) theo ngữ nghĩa toàn ảnh sau diffusion, rồi kiểm tra những cặp có đúng một ảnh đúng lớp. Ext-1 có 878 cặp như vậy ở 216 truy vấn, Ext-2 có 2.105 cặp ở 263 truy vấn. Độ chính xác cân bằng cho mỗi sketch lấy trung bình giữa giữ nguyên quyết định đúng và sửa quyết định sai của điểm toàn ảnh. Có lần lượt 59 / 157 sketch đủ cả hai loại để tính chỉ số này.

| Điểm dùng để xếp ảnh | Ext-1 mAP@all / P@100 | Ext-2 mAP@200 / P@200 | Đúng ở cặp sát hạng, cân bằng (%), Ext-1 / Ext-2 |
|---|---:|---:|---:|
| Ngữ nghĩa toàn ảnh sau diffusion | 80,563 / 80,272 | 72,862 / 71,531 | 50,00 / 50,00 theo cách định nghĩa |
| Candidate-Conditioned Regional Matching gốc | 78,302 / 76,826 | 68,414 / 67,156 | 62,38 / 64,17 |
| Dự báo vùng có hiệu chỉnh theo ba crop sketch | 4,616 / 6,766 | 9,963 / 5,801 | 54,35 / 55,79 |
| Cùng dự báo, xoay tọa độ vùng ảnh | 4,450 / 6,132 | 8,929 / 5,402 | 56,81 / 57,06 |
| Dự báo đã hiệu chỉnh, sau diffusion RGB | 6,304 / 2,451 | 6,311 / 3,908 | 55,56 / 55,85 |

Bộ đọc vùng gốc sửa 289 và làm hỏng 144 quyết định trên Ext-1, con số tương ứng là 617 và 422 trên Ext-2. Dự báo có hiệu chỉnh sửa 257 / 544 và làm hỏng 189 / 480. Chiều đúng về tọa độ không thắng đối chứng xoay trên cả hai split. Ước lượng kiểu lược thông tin theo từng sketch giúp một số phép dự báo ảnh mẫu, nhưng không tạo độ phân biệt lớp đủ mạnh để xếp cả gallery. Hai endpoint đo những năng lực khác nhau. [Protocol](protocol.md) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [Kết quả ghép cặp](paired-T4.zip)
