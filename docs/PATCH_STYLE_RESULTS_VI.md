# Kết quả âm của chuẩn hóa patch

| Precision (%) | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin |
|---|---:|---:|---:|
| CLIP gốc | 41,64 | 33,85 | 46,26 |
| Chuẩn hóa mean và variance của từng ảnh | 4,21 | 5,01 | 4,21 |

Với 16 sketch kiểm tra code, đường tính embedding gốc khớp CLIP chuẩn. Dưới biến đổi affine tổng hợp theo kênh, khoảng cách cosine trung bình giữa hai đầu ra giảm từ 0,2130 xuống khoảng 0,000018. Tuy nhiên sự bất biến này loại bỏ thông tin cần cho việc phân biệt ảnh. Trên Ext-1, ảnh sai top 1 của CLIP gần sketch hơn ảnh đúng tốt nhất theo khoảng cách moment chỉ trong 2,81% lỗi. Trên Ext-2 là 23,44%. Ở TU là 44,38%. Kết quả bác bỏ giả thuyết rằng chuẩn hóa toàn bộ moment tại vị trí này giải quyết được lỗi truy hồi.

Hàm gốc trong KD-SBIR được gọi lại trên toàn bộ embedding của cấu hình chính. Không có tham số được chọn lại sau khi xem điểm. [Giao thức](PATCH_STYLE_PROTOCOL_VI.md) · [Hai split Sketchy](results/patch-style-invariance/sketchy-complete.json) · [TU-Berlin](results/patch-style-invariance/tuberlin-complete.json)
