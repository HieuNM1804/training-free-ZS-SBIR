# Bảo toàn thứ tự giữa các patch

Mỗi truy vấn được xử lý riêng. Cấu hình được chọn dùng tám neuron nhạy với dịch chuyển ở block 2, độ phân giải 64 patch cho mỗi phân phối điều kiện, rồi lấy phân bố hạng điển hình từ 16 ảnh gallery gần truy vấn. Các từ công khai đọc ra thay đổi điểm truy hồi. Cả hai split unseen tham gia chọn cấu hình, với 36 sketch mỗi lớp và toàn gallery. Tất cả con số dưới đây là mAP / precision (%) qua evaluator KD-SBIR trên Colab T4.

| Phương án | Sketchy Ext-1, 900 truy vấn | Sketchy Ext-2, 756 truy vấn |
|---|---:|---:|
| Nền | 84,64 / 83,94 | 80,64 / 78,38 |
| Vận chuyển hạng theo ngữ nghĩa patch và cảnh | 84,63 / 84,18 | 80,93 / 78,55 |
| Dùng cùng một phân bố hạng tham chiếu cho mọi cảnh | 84,67 / 84,26 | 80,87 / 78,56 |
| Khớp histogram thông thường | 84,73 / 84,12 | 80,69 / 78,49 |
| Xoay thứ tự hạng sang patch khác | 84,73 / 84,09 | 80,72 / 78,31 |
| Thay bằng trung bình có điều kiện | 84,74 / 84,18 | 80,77 / 78,40 |

Các đối chứng được dò tham số riêng. So với khớp histogram, chênh mAP của đề xuất trên Ext-1 là −0,10 [−0,36; +0,12] điểm, trên Ext-2 là +0,24 [−0,04; +0,57] theo bootstrap lớp. So với nền, phương án chính sửa sáu top-1 và làm hỏng năm ở mỗi split. Các kết quả này chưa cho thấy việc chọn phân bố hạng theo cảnh là thành phần cần thiết.

Phép thử camera chỉ dùng ảnh thật trong hai nhóm giữ lại, mỗi nhóm 144 ảnh. Đo trên tám neuron nhạy với nhiễu độ sáng ở block 2. Khi thay đổi tương phản, chênh lệch bình phương chuẩn hóa trung vị giữa hoạt hóa và ảnh gốc là 0,018 trên Ext-1 và 0,024 trên Ext-2. Sau can thiệp, các số tương ứng tăng lên 0,274 và 0,272. Với thay đổi gamma, sai số cũng tăng trên cả hai nhóm. Ngay khi trực tiếp tạo một biến đổi tăng trong không gian phân vị, phương pháp khôi phục hạng tương đối nhưng có thể gán phân vị tuyệt đối sai. Sai số trung vị của biến đổi đã biết là 0,149 và 0,155; sau hiệu chỉnh là 0,257 và 0,227. Thay hạng sang vị trí khác làm sai số lớn hơn nhiều. Hai kết quả cùng chỉ ra vấn đề ở việc ước lượng phân bố hạng ban đầu của một ảnh mới.

Các kiểm tra CDF thu lại phân vị ban đầu với sai số tối đa dưới 0,000001 trên các giá trị được tách đủ xa. Khi các giá trị gần nhau ở độ chính xác FP32, có 0,26% giá trị trong mẫu kiểm tra đổi hạng với mức tối đa bằng nửa bậc hạng trong 49 patch. Chưa chạy chuyển giao TU-Berlin cho biến thể này. Chứng cứ hiện tại vẫn giới hạn ở hai split sketch → photo và các thay đổi camera có đối chứng. [Cơ chế, giả thuyết và nguồn paper](../../CONDITIONAL_PATCH_RANK_TRANSPORT.md)
