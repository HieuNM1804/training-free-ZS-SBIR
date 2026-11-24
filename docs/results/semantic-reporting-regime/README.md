# Có thể dự báo khi nào một ảnh sẽ giải thích vùng sketch chưa thấy?

Xét hai giả thuyết cho bốn crop tách biệt. Ở giả thuyết đầu, một crop sketch cung cấp phân bố từ cho cả các crop khác. Ở giả thuyết sau, ảnh ứng viên cho biết phân bố đó thay đổi thế nào khi chuyển vị trí. Từ độ khớp ở hai crop đã quan sát, tính hậu nghiệm Bayes của hai giả thuyết và dự báo crop để riêng. Bản vẽ khác của cùng ảnh chỉ dùng để chấm sau khi đã tạo dự báo cho từng sketch độc lập.

| Dự báo crop của bản vẽ khác cùng ảnh, mô hình Bayes trừ đối chứng (nat) | Ext-1 | Ext-2 |
|---|---:|---:|
| Log score ảnh đúng so với hỗn hợp xác suất cố định | −0,03773 | −0,02623 |
| Biên ảnh đúng trừ đối thủ cùng lớp so với hỗn hợp cố định | −0,03381 | −0,03849 |
| Log score ảnh đúng so với một crop sketch làm tham chiếu | −0,16243 | −0,14747 |
| Log score ảnh đúng so với lấy xác suất hai giả thuyết từ ảnh cạnh tranh | −0,03505 | −0,03749 |

Hai tập có 150 / 126 cặp ảnh cùng lớp tương ứng với 1.676 / 1.528 sketch. Khoảng bootstrap 95% theo cặp cho hàng đầu là [−0,04866, −0,02688] / [−0,03747, −0,01594]. Ở đây việc một ứng viên giải thích được hai crop không dự báo tốt rằng cách giải thích ấy tiếp tục đúng cho crop mới. Tiêu chí xác nhận mô hình chế độ chung đã thất bại. Đơn vị là log score dự báo phân bố từ CLIP, không có diễn giải trực tiếp thành mAP. [Giả thuyết và các đối chứng](protocol.md) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [Đầu ra theo ca](paired-T4.zip)
