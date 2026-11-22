# Tách xác suất chú ý đến danh từ và thông tin mà danh từ truyền đi

Một attention head có thể đọc danh từ nhiều hơn hoặc đọc thông tin khác từ cùng danh từ. Ở lớp cuối của text encoder, chúng tôi đo hai đại lượng riêng dưới hai mô tả “sketch” và “photo”, hoán đổi từng đại lượng rồi chạy qua phần còn lại của CLIP. Hai hiệu ứng cộng lại khớp với can thiệp đồng thời, sai số lớn nhất 1,49 × 10⁻⁸. Trên 20.931 tên công khai, xác suất chú ý vào các token của danh từ trung bình chuyển từ 0,273 sang 0,282. [Cơ chế, các đối chứng và nguồn](../../NOUN_ATTENTION_TRANSFER.md)

| mAP / precision (%) trên mẫu 36 sketch mỗi lớp | Sketchy Ext-1 | Sketchy Ext-2 |
|---|---:|---:|
| Mốc khớp từ và vùng | 84,64 / 83,96 | 80,63 / 78,38 |
| Tách hai đường truyền, chọn hai hệ số | 84,99 / 84,60 | 80,97 / 78,52 |
| Một hệ số cho toàn bộ biến thiên, chọn riêng | 84,79 / 84,08 | 81,16 / 78,65 |
| Chỉ đổi xác suất chú ý, chọn riêng | 84,59 / 83,87 | 80,70 / 78,43 |
| Chỉ đổi nội dung đọc được, chọn riêng | 84,94 / 84,47 | 81,01 / 78,60 |
| Hoán vị thay đổi xác suất giữa các head, cùng hệ số với đề xuất | 85,02 / 84,53 | 80,99 / 78,39 |
| Dùng xác suất từ phía nguồn, cùng hệ số với đề xuất | 85,04 / 84,62 | 81,00 / 78,53 |

Cấu hình chính điều chỉnh biểu diễn toàn sketch ở lớp 11, với hệ số −5 cho biến thiên xác suất và −10 cho biến thiên nội dung. Khi hoán vị các head, mAP vẫn giữ được trên cả hai split. Như vậy, phép tách có thể đo được bằng forward pass nhưng phân bố thay đổi theo đúng head chưa giải thích được lợi ích truy hồi. Hai split gồm 900 và 756 truy vấn với gallery đầy đủ. Cấu hình cuối và đối chứng được chấm qua evaluator gốc. Nhánh dừng ở pilot Sketchy. [Kết quả, lựa chọn và kiểm tra KD-SBIR](pilot.json) · [Số liệu từng truy vấn](pilot-per-query.zip)
