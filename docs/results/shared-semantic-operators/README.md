# Liệu cấu trúc chung trong gallery có giúp truy hồi xuyên miền?

Mỗi con số trong bảng là mAP / precision, đơn vị phần trăm, đo bằng KD-SBIR trên T4. Cấu hình dùng 32 chiều, toán tử thăm dò từ vùng ảnh, mức phạt 16 và hệ số gộp 0,35. Cấu hình cùng 14 đối chứng được khóa sau hai split Sketchy unseen rồi chấm trên đủ 30 lớp TU-Berlin. Cả ba cột đều dùng truy vấn đầy đủ. Ext-1 và TU dùng mAP@all / P@100, Ext-2 dùng mAP@200 / P@200.

| Phương án | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin |
|---|---:|---:|---:|
| Nền từ điển, vùng và gallery RGB | 84,37 / 83,85 | 77,02 / 74,83 | 80,07 / 80,92 |
| Cấu trúc chung, thăm dò qua vùng ảnh | 84,32 / 83,89 | 77,56 / 75,56 | 79,24 / 80,53 |
| Thăm dò qua từ công khai | 84,30 / 83,94 | 77,58 / 75,66 | 79,24 / 80,57 |
| Xoay đồng thời các trục, tự chọn cấu hình | 84,38 / 83,98 | 77,27 / 75,26 | 79,63 / 80,81 |
| Gộp trực tiếp, tự chọn cấu hình | 84,36 / 83,93 | 77,21 / 75,16 | 79,67 / 80,82 |

Khoảng tin cậy 95% của chênh lệch mAP với nền, bootstrap ghép cặp theo lớp, là [−0,40; +0,28] điểm trên Ext-1, [−0,80; +2,18] trên Ext-2 và [−1,67; −0,09] trên TU. Trên Ext-2 phương án sửa 331 lỗi top-1 và tạo 105 lỗi mới. Trên TU các số tương ứng là 41 và 44. Các điểm mAP và precision của mọi cấu hình cuối đều được đối soát qua evaluator gốc với sai số tối đa dưới 0,000001 trên từng truy vấn.

Thử nghiệm kiểm tra xem thăm dò chung trên nhiều ảnh có phân biệt quan hệ ngữ nghĩa ổn định với tương quan do gallery hay không. Trên pilot Ext-2, với hệ số 0,15, trung bình qua ba lần chia gallery đạt 80,945 khi dùng đúng liên hệ, 80,702 khi tráo và 80,909 khi thay dữ liệu bằng trung bình cụm. Hiệu ứng sát đối chứng cụm. Khi dùng nhãn lớp chỉ để chẩn đoán, lọc tín hiệu đã làm chất lượng mAP đọc lại lớp giảm trên cả hai split. Kết quả hiện có chưa xác nhận phần riêng của toán tử đủ mạnh để theo đuổi như đóng góp chính.

Thư mục `integrated` chứa cấu hình đã khóa, JSON đầy đủ, kết quả từng truy vấn được nén theo split, các trường hợp tráo và thông tin đối soát. Hai split Sketchy đã tham gia chọn tham số. TU đo chuyển giao giữa bộ dữ liệu trong cùng chiều sketch → photo. [Giả thuyết, cơ sở lý thuyết và công trình tham khảo](../../SHARED_SEMANTIC_OPERATORS.md)
