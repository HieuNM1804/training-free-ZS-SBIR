# Cùng một vùng ảnh có mang cùng tín hiệu khi thay cách đọc CLIP?

Giữ tọa độ bốn góc, từ vựng cục bộ và ảnh tham chiếu cố định. Khi dự báo một vùng, cách đọc *cùng tọa độ* dùng phân bố từ ứng viên tại chính vị trí đang được giữ lại. Hai câu hỏi khác nhau là liệu đây có phải vị trí đúng về nghĩa và liệu CLIP có cho cùng loại thông tin khi đổi cách lấy đặc trưng. Các ảnh tự so với chính chúng cung cấp đối chứng dương cho câu hỏi thứ hai.

| Nguồn → ứng viên | Cách đọc vùng ứng viên | Ghép đúng nguồn Ext-1 / Ext-2, cùng tọa độ | Ghép đúng nguồn Ext-1 / Ext-2, điều kiện theo một vùng đã thấy |
|---|---|---:|---:|
| Sketch → ảnh thật | Token toàn ảnh | 41,31% / 42,63% | 43,97% / 47,79% |
| Sketch → ảnh thật | Crop mã hóa riêng | 69,75% / 66,69% | 63,18% / 63,10% |
| Ảnh thật → chính ảnh đó | Token toàn ảnh | 36,67% / 39,29% | 59,67% / 59,52% |
| Ảnh thật → chính ảnh đó | Crop mã hóa riêng | 100% / 100% | 100% / 100% |

Với crop mã hóa riêng ở hai miền, lợi ích của bố cục thật so với xáo tọa độ khi dự báo vùng khác tăng chênh log score ảnh đúng trừ ảnh đối thủ **0,00967** và **0,00786 nat**. Khoảng 95% theo cặp là [0,00369, 0,01628] và [0,00312, 0,01275]. Tọa độ cố định trên ảnh gốc cho hiệu ứng còn lớn hơn. Đây là tín hiệu để kiểm tra trên ảnh mới, với tọa độ phải được suy ra từ nội dung. Các bản vẽ có cùng nguồn chỉ gặp nhau ở phép đánh giá ghép nguồn.

[Thiết kế và ý nghĩa của đối chứng ảnh tự so khớp](../regional-predictive-observation/observation-control.md) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [Số liệu theo truy vấn](paired-T4.zip)
