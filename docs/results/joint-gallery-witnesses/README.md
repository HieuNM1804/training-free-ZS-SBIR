# Một nhóm ảnh cùng xác nhận một phân bố từ

Với mỗi ứng viên, liệt kê các nhóm láng giềng bao quanh nó trong mặt phẳng tiếp tuyến CLIP. Trong từng nhóm, lấy phần phân bố từ mà mọi ảnh trong nhóm cùng hỗ trợ. Chấm phần phân bố chung bằng Dot đủ 601 từ và chọn nhóm hợp nhất với truy vấn. Điểm cuối là cực đại giữa điểm trực tiếp và bằng chứng từ nhóm. Một nhóm xác nhận được dùng chung cho toàn bộ vector từ. [Công thức, cơ sở lý thuyết và prior art](protocol.md)

## Kết quả

| Cấu hình | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin 30 lớp |
|---|---:|---:|---:|
| Dot đủ 601 từ | 75,35 / 77,10 | 72,66 / 71,08 | 73,46 / 73,98 |
| **Một nhóm xác nhận phần phân bố chung** | **76,24 / 77,21** | **72,78 / 71,28** | **74,69 / 74,16** |
| Bao lồi sau khi tính Dot | 76,76 / 77,34 | 73,03 / 71,41 | 75,13 / 74,32 |
| Bao lồi riêng từng từ | 76,66 / 77,53 | 72,86 / 71,46 | 75,03 / 74,46 |
| RGB diffusion trên cùng graph | 78,44 / 78,67 | 74,63 / 72,66 | 76,75 / 75,33 |

Mỗi ô là mAP / precision (%). Ext-1 và TU-Berlin dùng mAP@all / P@100. Ext-2 dùng mAP@200 / P@200. Đã chấm đủ 14.179 / 12.694 / 2.130 query và 17.100 / 12.552 / 27.262 ảnh gallery bằng KD-SBIR gốc. Lượt mới chạy trên Colab T4, commit 1f1a635, seed 42, OpenAI CLIP ViT-B/32 đóng băng, cùng cấu hình cho cả ba split. Các đối chứng lấy từ lượt đã xác minh trên cùng dữ liệu, checkpoint và thứ tự ứng viên. Precision của Dot và hai phép bao lồi được tái lập khi tính đối chứng ghép cặp. [Khóa cấu hình](freeze.json) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU-Berlin](tuberlin.json) · [Kết quả từng truy vấn](paired-T4.zip)

## Kiểm tra cơ chế

Trong phép dựng có cùng điểm Dot và cùng tọa độ lân cận, chứng nhận chung phân biệt trường hợp hai từ chỉ được hỗ trợ ở hai phía với trường hợp một từ được hỗ trợ quanh ứng viên. Khi hai từ mỗi bên có tam giác riêng, phép bao riêng từng từ nâng điểm lên 0,89, còn phép chứng nhận lấy một nhóm cho toàn bộ vector cho 0,45. Cận trên từ nhánh sau Dot được kiểm tra trên mọi cặp truy vấn–ảnh thực, với độ vi phạm tối đa bằng 0. Bộ liệt kê khớp phép vét cạn trên 32 chart mỗi split. Kiểm tra riêng trên T4 với bốn hướng vuông góc tìm đúng nhóm bốn ảnh, và với toàn bộ nhân chứng cùng hướng trả mức hỗ trợ 0.

Ở cùng ngưỡng điểm của top-K Dot, những ứng viên được nhánh chung giữ lại có tỷ lệ cùng lớp cao hơn nhóm chỉ qua ngưỡng theo nhánh sau Dot. Nhánh chung cũng đưa ít ảnh mới vào hơn. Để kiểm tra chất lượng lựa chọn, lấy cùng số ảnh mới xếp cao nhất của từng nhánh trên mỗi truy vấn.

| Đúng lớp trong ảnh mới đã ghép số lượng | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin |
|---|---:|---:|---:|
| Phân bố chung / sau Dot | 87,70 / 87,72 | 79,68 / 79,83 | 86,22 / 86,23 |
| Phân bố chung / từng từ | 87,72 / 87,78 | 79,68 / 79,90 | 86,24 / 86,11 |

Chênh lệch theo lớp so với nhánh sau Dot là −0,12 / −0,42 / −0,07 điểm, với khoảng bootstrap 95% đều chứa 0. Đối chứng ghép số lượng chưa xác nhận giả thuyết rằng một phân bố được chứng nhận chung chọn đúng ảnh tốt hơn. Tăng mAP so với Dot là +0,90 / +0,13 / +1,24 điểm. Cơ chế có sai khác đại số với diffusion và hai phép bao lồi đối chứng. Bằng chứng truy hồi hiện chỉ xác nhận việc dùng gallery có ích trên hai benchmark trong cùng chiều sketch → photo.

Sau khi có embedding và posterior, lập chỉ mục và chấm GPU mất 11,02 / 7,78 / 3,96 giây cho ba split. Việc chấm phải xét lần lượt 1,99 / 1,47 / 3,20 triệu nhóm tam giác trong các gallery, tốn phép tính hơn Dot trên một vector mỗi ảnh. Chạy theo block 1.024 ảnh và batch 768 truy vấn cho phần nhân ma trận.
