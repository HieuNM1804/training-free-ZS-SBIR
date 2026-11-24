# Kích thước, vị trí và tương tác khi chọn vùng cho từng ứng viên

Mỗi ô là mAP / precision (%). Chấm 1.536 sketch mỗi split với toàn bộ gallery bằng KD-SBIR gốc trên Colab Tesla T4, seed 42. Các hàng đều có cùng điểm ngữ nghĩa toàn ảnh và cùng hệ số vùng. GCWE được tắt để đo trực tiếp cơ chế vùng.

| Cách dùng bằng chứng vùng | Ext-1 · mAP@all / P@100 | Ext-2 · mAP@200 / P@200 |
|---|---:|---:|
| Gốc, ứng viên tự chọn cả kích thước và vị trí | 84,658 / 83,796 | 77,325 / 75,036 |
| Chỉ kích thước phụ thuộc ứng viên | 84,559 / 83,669 | 76,499 / 74,186 |
| Chỉ vị trí phụ thuộc ứng viên | 84,343 / 83,462 | 77,248 / 74,909 |
| Cộng hai hiệu ứng chính | 84,657 / 83,751 | 77,586 / 75,199 |
| Kích thước cộng tương tác, bỏ hiệu ứng chính vị trí | 84,540 / 83,693 | 76,410 / 74,032 |
| Kích thước cộng tương tác từ ứng viên khác | 84,531 / 83,658 | 76,504 / 74,299 |

Biến thể chính ở hàng áp chót giảm 0,118 / 0,915 mAP so với gốc và thua cả phương án chỉ chọn kích thước trên hai split. Không đưa biến thể này sang TU. Hàng cộng hai hiệu ứng chính là kết quả thăm dò, đạt −0,001 / +0,261 mAP so với gốc. Đã khóa phép xác nhận trên 3.072 truy vấn khác mỗi split trước khi quyết định có phát triển tiếp.

Kiểm tra lần đầu dùng điểm lưu trên L4, với giới hạn 0,04 điểm phần trăm cho sai lệch tuyệt đối trung bình của mỗi metric trên từng truy vấn. Ext-2 vượt giới hạn này (0,05627 AP và 0,04850 precision). Giữ nguyên công thức, đầu vào và tham số, chạy lại score tham chiếu gốc trên T4 theo batch 16 và so với pilot batch 24. Sai lệch AP / precision trung bình tuyệt đối cùng T4 là 0,000014 / 0,000000 ở Ext-1 và 0,000129 / 0,000326 điểm ở Ext-2. Giá trị trung bình không xác định nguyên nhân của từng khác biệt phần cứng.

[Giao thức đã khóa](protocol.md) · [Kết quả Ext-1](sketchy_1.json) · [Kết quả Ext-2](sketchy_2.json) · [Theo từng truy vấn](paired-T4.zip) · [Giao thức xác nhận](../regional-additive-confirmation/protocol.md)
