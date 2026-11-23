# Tên của nhóm ảnh có cần được kiểm tra lại ở chính ảnh ứng viên?

Trên Colab L4, chấm pilot của hai split Sketchy với 36 sketch mỗi lớp và toàn gallery. Hai split có 900 và 756 truy vấn. Các hệ số đều được chọn thăm dò trên hai pilot, cho cả phương pháp và các đối chứng. Mỗi ô là mAP / precision theo %, Ext-1 dùng mAP@all / P@100 còn Ext-2 dùng mAP@200 / P@200. 35 cấu hình chọn lọc được phát lại bằng evaluator KD-SBIR gốc, sai số tối đa 2,39 × 10⁻⁷. [Mô hình và các giả thuyết](../../SHARED_PRIVATE_NAME_EVIDENCE.md)

| Cấu hình | Sketchy Ext-1 | Sketchy Ext-2 |
|---|---:|---:|
| Posterior tên trên cây, xác minh từ ở chính ảnh | 80,25 / 81,96 | 76,76 / 74,28 |
| Giữ nguyên từng xác suất mượn tên và cùng posterior nhóm, bỏ xác minh danh tính từ | 80,37 / 81,99 | 77,17 / 74,31 |
| Prototype với xác minh từ, chọn tham số riêng | 80,83 / 82,25 | 77,01 / 75,03 |
| Prototype với hệ số mượn tên mỗi ảnh, chọn tham số riêng | 81,24 / 82,48 | 77,34 / 74,45 |
| Nền vùng/từ + diffusion | 84,64 / 83,96 | 80,63 / 78,38 |
| Chỉ thêm phần hiệp phương sai được xác định từ mô hình cây vào nền vùng/từ | 84,64 / 83,96 | 80,64 / 78,39 |

Giữ nguyên hệ số tại từng ảnh, phần xác minh danh tính từ làm mAP giảm 0,130 điểm trên Ext-1 và 0,410 trên Ext-2. Khoảng bootstrap 95% theo lớp lần lượt từ −0,224 đến −0,032 và từ −0,794 đến −0,062. Đã dừng nhánh này ở pilot. Mỗi ảnh và mỗi truy vấn đều dùng đủ 20.931 từ, ảnh ứng viên và ảnh ước lượng tên nhóm thuộc hai nửa gallery rời nhau.

## Điều tìm thấy trên lỗi thật

Dùng tên lớp sau khi đã xếp hạng để kiểm tra lý do. Trên Ext-1, trong 122 lỗi top-1 mà cả nhãn đúng và nhãn sai có tên tương ứng, toàn sketch chỉ cho tên đúng điểm cao hơn tên đối thủ ở 31 trường hợp. Trên Ext-2 là 24/80. Nếu chỉ so với các tên lớp có mặt, số truy vấn xếp đúng tên đầu tiên là 732/864 và 540/648. Các số này là chẩn đoán có nhãn, chỉ bao phủ những lớp có tên tương ứng trong từ điển.

Xét ảnh đúng và ảnh sai đứng cao nhất của nền vùng/từ ở các truy vấn có top-1 sai. Trên Ext-1, hệ số mượn tên trung bình của ảnh sai là 0,751, của ảnh đúng là 0,648. Trên Ext-2 các giá trị là 0,739 và 0,637. Phần xác minh làm số cặp có khoảng cách điểm đúng dấu giảm từ 27 xuống 26 và từ 28 xuống 26. Sai số còn lại của đồng nhất thức hiệp phương sai là dưới 4 × 10⁻¹⁶ sau khi tách phần làm tròn của xác suất FP32. Chất lượng của tín hiệu phân biệt hai cách hiểu sketch là một hạn chế cụ thể mà việc làm hậu nghiệm gallery sắc hơn chưa giải quyết được.
