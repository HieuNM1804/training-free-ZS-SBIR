# Đóng góp của bốn thành phần trong cấu hình đạt 85,48 mAP trên Ext-1

Mỗi lượt tắt một thành phần rồi chấm lại tất cả truy vấn. Mỗi ô là mAP / precision (%) và số trong ngoặc là thay đổi mAP so với cấu hình đủ bốn thành phần, tính trước khi làm tròn.

| Cấu hình | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin |
|---|---:|---:|---:|
| Đủ bốn thành phần | 85,48 / 84,77 | 78,14 / 76,00 | 81,69 / 82,46 |
| Bỏ diffusion trên gallery RGB | 82,66 / 84,54 (-2,83) | 76,69 / 75,37 (-1,45) | 79,51 / 81,34 (-2,18) |
| Bỏ điểm từ các vùng của truy vấn | 83,99 / 83,22 (-1,49) | 76,25 / 74,77 (-1,89) | 81,36 / 82,21 (-0,33) |
| Bỏ điểm theo phân cấp gallery | 84,82 / 84,26 (-0,66) | 78,07 / 75,78 (-0,07) | 81,39 / 82,11 (-0,30) |
| Bỏ cập nhật từ bằng hình học patch | 85,29 / 84,65 (-0,19) | 77,56 / 75,46 (-0,58) | 80,70 / 81,49 (-0,98) |

Ext-1 dùng mAP@all và P@100 cho 14.179 sketch trên 17.100 ảnh. Ext-2 dùng mAP@200 và P@200 cho 12.694 sketch trên 12.552 ảnh. TU-Berlin dùng mAP@all và P@100 cho 2.130 sketch trên 27.262 ảnh, đủ 30 lớp unseen. Cả năm dòng dùng cùng đặc trưng đã dựng lại trên Colab Tesla T4, cùng hệ số đã khóa, cùng từ điển Open Images và evaluator KD-SBIR gốc. Chênh lệch tuyệt đối lớn nhất giữa hai metric trung bình của dòng đầy đủ và số đã công bố trước đây là 0.00088 điểm phần trăm. [Protocol](protocol.md)

- Diffusion lan truyền điểm qua các ảnh gallery gần nhau trong không gian CLIP RGB. Lượt bỏ tắt cả lan truyền điểm từ toàn ảnh và lan truyền trọng số 64 từ của nhánh vùng. Khoảng bootstrap 95% theo lớp của thay đổi mAP khi bỏ diffusion nằm dưới 0 trên cả ba split.
- Điểm đọc vùng đóng góp nhiều nhất trên Ext-2. Khi tắt, hệ số 0,35 của điểm đọc vùng được đưa về 0. Nhánh phân cấp vẫn đọc cùng các crop của nó và tự tính độ lệch chuẩn từ điểm còn lại theo công thức đã khóa.
- Phân cấp có đóng góp trung bình nhỏ trên Ext-2 và TU. Khoảng bootstrap 95% theo lớp khi bỏ phân cấp chứa 0 trên cả ba split.
- Cập nhật từ bằng hình học có đóng góp lớn nhất trên TU trong ba split. Ở đối chứng trước đây chỉ dùng khớp từ trực tiếp, thêm hình học tăng 0,21 / 0,69 / 1,28 mAP tương ứng trên Ext-1 / Ext-2 / TU. [Đối chứng khớp từ trực tiếp](../geometry-conditioned-word-evidence/README.md)

Mức thay đổi là hiệu ứng có điều kiện khi giữ ba phần còn lại. Cả bốn phép bỏ đều làm giảm mAP trung bình ở cả ba split. Tương tác riêng giữa một cặp thành phần cần đối chứng bật/tắt cả hai yếu tố. Bootstrap lấy mẫu lại các lớp với trọng số bằng nhau, còn metric chính lấy trung bình theo truy vấn. Hai split Sketchy có dữ liệu chồng lặp và cùng tham gia phát triển các cấu hình trước khi khóa sang TU.

Mã chạy `03f85754faace137ee1e7c47d903da294b8df49a`. Các JSON chứa cả hai metric, hai độ lệch theo cặp, bootstrap theo lớp và kiểm tra với kết quả cũ. Tệp ZIP lưu cả AP và precision của từng truy vấn, nhãn đánh giá và thứ tự gallery cho cả ba split. [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU-Berlin](tuberlin.json) · [Kết quả theo truy vấn](paired-t4.zip)
