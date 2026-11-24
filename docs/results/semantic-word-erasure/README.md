# Tách xác suất xuất hiện của từ và vị trí xuất hiện

Phép thử trước bắt mỗi từ phải có một vị trí ngay cả khi chỉ có rất ít bằng chứng từ đó xuất hiện. Ở đây chọn một vùng đều nhau, rút mô tả từ phân bố CLIP trên 20.931 từ, rồi quan sát một trong năm kết quả cho từ đang truy vấn. Bốn kết quả chỉ vị trí, kết quả thứ năm là từ không được quan sát. Nếu sketch chỉ dịch chuyển hoặc xóa bằng chứng đã có trên ảnh, một phép biến đổi chung phải giải thích được các quan sát của nhiều từ. Đã kiểm tra các bất đẳng thức cần cho giả thuyết đó trên 768 sketch mỗi split và gallery đầy đủ.

| Đại lượng dùng để xếp ảnh | Ext-1 mAP@all / P@100 | Ext-2 mAP@200 / P@200 |
|---|---:|---:|
| Ngữ nghĩa toàn ảnh gốc | 80,563 / 80,272 | 72,862 / 71,531 |
| Kiểm tra từng từ có bị khuếch đại khi sang sketch | 17,489 / 18,238 | 20,144 / 17,971 |
| Thêm cặp từ và xét cả bốn vị trí | 18,046 / 19,415 | 21,820 / 19,073 |
| Thêm cặp từ, chỉ quan sát sự xuất hiện | 19,052 / 20,908 | 23,770 / 20,163 |
| Hoán vị riêng vị trí của từng từ bên ảnh | 18,345 / 19,751 | 22,110 / 19,401 |

Trong 0,98% / 1,89% các cặp sketch–ảnh, quan sát vị trí của hai từ siết được chặn dưới quá ngưỡng làm tròn 2×10⁻⁶ so với kiểm tra từng từ. Chênh lệch mAP so với cùng mô hình chỉ quan sát sự xuất hiện là **−1,005 / −1,950 điểm**. Khoảng bootstrap theo lớp của hai mức chênh lệch đều âm. Ở cả hai split, không có cặp nào vượt qua toàn bộ điều kiện trong sai số số học đã định. Vì thế bộ lọc dựa trên tính khả thi trả lại đúng thứ hạng ngữ nghĩa toàn ảnh.

Việc cho phép một từ vắng mặt giải quyết được vấn đề chuẩn hóa vị trí khi tạo quan sát. Tuy nhiên, giả thuyết cấm sketch tăng độ nổi bật của bất kỳ từ nào so với ảnh quá chặt đối với chính readout CLIP này. Kết quả chưa đủ để quy sự khuếch đại cho người vẽ; chênh lệch hiệu chuẩn theo miền của encoder cũng phù hợp với bằng chứng. [Giả thuyết, chứng minh và đối chứng](protocol.md) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [Kết quả từng truy vấn](paired-T4.zip)
