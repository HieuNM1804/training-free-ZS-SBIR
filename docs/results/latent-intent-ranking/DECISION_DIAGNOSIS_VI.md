# Vì sao sửa được nhiều lỗi ở hạng đầu trong khi mAP@200 giảm?

Truy hồi được chấm trên đủ 12.694 sketch của Ext-2 bằng cấu hình đã khóa. Nhãn chỉ được sử dụng sau khi chọn thứ hạng. Cột cuối đo số điểm mà từng nhóm đóng góp vào thay đổi mAP của toàn split.

| Điều xảy ra sau khi đổi thứ hạng | Số sketch | Đóng góp vào thay đổi mAP |
|---|---:|---:|
| Sửa kết quả sai ở hạng đầu | 153 | +0,152 |
| Mất kết quả đúng ở hạng đầu | 72 | −0,219 |
| Hạng đầu vẫn đúng và có đổi thứ hạng | 4.098 | +0,050 |
| Hạng đầu vẫn sai và có đổi thứ hạng | 980 | −0,173 |
| Giữ thứ hạng nền | 7.391 | 0 |
| **Tổng** | **12.694** | **−0,190** |

Khi xét số ảnh đúng trong top 200, 1.115 sketch có thêm trung bình 8,24 ảnh đúng và tăng trung bình 2,54 điểm AP. 464 sketch mất trung bình 17,18 ảnh đúng và giảm 10,49 điểm AP. Có 41 sketch từ có ít nhất một ảnh đúng chuyển thành mất toàn bộ ảnh đúng trong top 200. Chỉ 16 sketch đi theo chiều ngược lại. Chính độ lớn của thiệt hại trong nhóm nhỏ khiến mAP giảm dù nhiều truy vấn có thêm ảnh đúng.

Trong nhóm được thay đổi, tương quan Pearson giữa mức tăng AP dự báo theo gallery và mức tăng AP@200 thực tế là −0,061 trên Ext-2. Đại lượng tương ứng trên Ext-1 là +0,114. Một ngưỡng trên độ tự tin hiện có sẽ khó nhận diện các lần mất nhiều ảnh nếu thứ tự độ tự tin này tiếp tục kém trên dữ liệu khác. Đây là phân tích hậu nghiệm trên Sketchy. [Số liệu và các khoảng theo độ tự tin](decision-diagnosis.json) · [Bảng đầy đủ với TU-Berlin](README.md)
