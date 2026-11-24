# Xác nhận với một sketch trên mỗi ảnh nguồn chưa dùng trong phép thử chọn vùng

Khóa quy tắc chọn ảnh và tính điểm trước khi chạy. Loại mọi ảnh nguồn đã cung cấp sketch cho tập thăm dò ở một trong hai split. Mỗi ảnh nguồn còn lại cung cấp một sketch theo seed 42. Hai lượt chấm dùng toàn bộ gallery tương ứng. Do hai split Sketchy có thể trùng ảnh, mức độ tái lập được diễn giải riêng cho từng split. Quy tắc chọn cặp chặt làm số truy vấn đủ điều kiện thấp hơn rõ rệt so với số đã chấm.

| Chỉ số | Ext-1 | Ext-2 |
|---|---:|---:|
| Sketch từ các ảnh nguồn mới | 1.764 | 1.389 |
| Cặp có đủ hai loại ảnh chọn vùng / tổng cặp sát hạng đúng–sai | 504 / 1.875 | 2.340 / 4.036 |
| Sketch có cặp hợp lệ / có đủ hai hướng ban đầu | 81 / 48 | 237 / 175 |
| Tăng log score của ảnh đúng khi dùng ảnh cùng lớp thay ảnh khác lớp | +0,3305 | +0,0994 |
| Tăng log score của ảnh sai | +0,1955 | +0,0457 |
| Tăng biên đúng trừ sai, trung bình theo sketch | +0,1350 | +0,0536 |
| Độ chính xác cân bằng khi dùng ảnh cùng lớp | 57,58% | 62,31% |
| Độ chính xác cân bằng khi dùng ảnh khác lớp | 60,68% | 61,20% |
| Chênh lệch theo sketch (điểm %) | **−3,10** | **+1,11** |

Khi cho trọng số bằng nhau cho các lớp, khoảng bootstrap 95% của mức tăng log score ở ảnh đúng và ảnh sai đều dương trên cả hai split. Khoảng của **chênh lệch biên** là [−0,1308, 0,2694] / [−0,00008, 0,2098]. Với thay đổi độ chính xác cân bằng, khoảng theo lớp là [−23,40, 3,91] / [−9,76, 19,94] điểm. Vì vậy bằng chứng ủng hộ việc chuyển cách chọn vùng giữa hai ảnh cùng lớp ở mức điểm riêng của ảnh. Lợi thế phân biệt đúng và sai theo lớp chưa tái lập. Các nhóm ảnh chọn vùng có thể còn khác nhau về độ tương đồng thị giác; thí nghiệm khống chế khoảng cách và hướng thứ hạng. Nhãn chỉ phục vụ phân nhóm để chẩn đoán. [Thiết kế khóa trước](protocol.md) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [Các cặp, điểm và ID để tái lập](paired-T4.zip)
