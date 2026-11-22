# Kiểm tra bằng chứng còn đứng vững khi hoàn thiện bản vẽ

Trên Colab T4, mỗi sketch được chia thành sáu nhóm patch gốc theo ba cách. Đánh giá cả 64 tập con, gồm ảnh đủ phần và ảnh với mọi patch nhận cùng token trung bình. Mức độ đủ của một tập giữ lại được tính đúng trên mọi tập mở rộng, rồi so với kỳ vọng trên sáu lần tráo quan hệ bao hàm. [Giao thức, định nghĩa và công trình liên quan](../../PERSISTENT_DRAWING_EVIDENCE.md)

Mỗi ô là mAP / precision (%) qua KD-SBIR. Pilot dùng 900 và 756 truy vấn, toàn bộ gallery. Hai split unseen tham gia chọn tham số. Đối chứng đơn giản tốt nhất giảm điểm ảnh đã nhận nhiều bằng chứng từ một phần riêng lẻ của sketch.

| Phương án | Sketchy Ext-1 | Sketchy Ext-2 |
|---|---:|---:|
| Nền vùng | 84,64 / 83,94 | 80,64 / 78,38 |
| Hiệu ứng của đúng quan hệ bao hàm | 84,74 / 83,64 | 81,10 / 78,65 |
| Giảm bằng chứng trung bình của từng phần, tự chọn cấu hình | 84,83 / 84,40 | 81,39 / 78,99 |
| Tráo quan hệ bao hàm, tự chọn cấu hình | 84,50 / 83,74 | 81,27 / 78,94 |
| Chênh lệch giữa hai nhóm tráo | 84,47 / 83,67 | 81,22 / 79,03 |

Phương án chính dùng ba nhóm trên lưới, đọc qua từ công khai, hệ số 0,06. Chênh mAP so với nền là +0,10 và +0,46 điểm. Khoảng tin cậy 95% theo lớp lần lượt là [−0,32; +0,48] và [−0,37; +1,40]. So với phép giảm bằng chứng trung bình của từng phần, chênh lệch là −0,09 và −0,29 điểm. Khi chỉ có số liệu này, câu hỏi đáng kiểm tra hơn là tại sao một ảnh nhận được điểm cao khi chỉ nhìn chi tiết rời của sketch. Quan hệ bao hàm chưa chứng minh được lợi ích tăng thêm, và chưa được đưa sang TU-Berlin.

Có thể đo trực tiếp khoảng cách giữa phép che nhân tạo và cách người thật vẽ khác nhau. Với mỗi sketch đủ dữ liệu, lấy các bản vẽ **khác** của cùng vật và tính độ phân tán embedding giữa chúng. Bốn hướng do thêm từng nhóm patch trên lưới giải thích 9,10% và 8,58% biến thiên này ở hai split. Nếu dùng sketch khác cùng lớp có AP tương đồng để tạo bốn hướng, con số lần lượt là 8,58% và 7,67%. Phần tăng trong độ đặc hiệu theo đối tượng do **đúng** quan hệ patch, sau đối chứng tráo trong từng kích thước, chỉ đạt 0,044 và 0,020 điểm phần trăm. Có 893 và 749 truy vấn đủ số bản vẽ tham chiếu, trung bình 4,98 và 5,32 bản vẽ khác mỗi truy vấn. [Chẩn đoán biến thiên giữa các bản vẽ thật](human-drawing-variance.json)

Cũng đã tính kết quả khi hai ảnh ứng viên gặp đúng các phần sketch được hé lộ giống nhau. Trên các cặp ảnh kề nhau ở những hạng xác định trước, đọc qua từ với lưới và hai phần cho độ đồng thuận với bản vẽ khác cùng đối tượng 54,3% / 55,0%. Đọc toàn sketch đạt 53,9% / 54,8%. Khi trừ tác động quan sát được với một vật khác cùng lớp, phần tăng còn −0,34 / +0,07 điểm. Nhãn và thông tin bản vẽ cùng ảnh chỉ được dùng trong các chẩn đoán sau xếp hạng. [Kết quả đối chiếu từng cặp ảnh](independent-drawing-margins.json)
