# Khi mở một vùng mới, bằng chứng hình học có chuyển sang một bản vẽ độc lập?

Dùng toàn bộ sketch đã che vùng B làm bối cảnh. Với đủ 20.931 từ, một phân bố từ toàn cảnh dự đoán B thông qua các ảnh gallery và bố cục của chúng. Sau khi thấy B, cập nhật phân bố từ bằng điều kiện hóa Jeffrey. Kiểm tra dự đoán trên các bản vẽ khác của cùng ảnh và trên chính ảnh tham chiếu. Hạng của mỗi truy vấn chỉ phụ thuộc vào một sketch. Đối với phép đo từ ở ảnh tham chiếu, ảnh đó bị loại khỏi phía gallery cung cấp bằng chứng.

| Cấu hình | Ext-1 mAP@all / P@100 | Ext-2 mAP@200 / P@200 |
|---|---:|---:|
| Bối cảnh che một vùng, trước cập nhật | 79,368 / 79,810 | 70,838 / 69,149 |
| Sau khi quan sát vùng và cập nhật có hình học | 79,597 / 79,967 | 69,941 / 68,281 |
| Bố cục dịch sai | 79,608 / 79,980 | 69,937 / 68,260 |
| Gán sai mô tả ngữ nghĩa giữa các ảnh gần hạng | 79,367 / 79,880 | 70,978 / 69,163 |
| Giới hạn 64 từ trước cập nhật | 78,440 / 78,776 | 70,461 / 68,479 |
| Giới hạn 64 từ sau cập nhật | 78,577 / 78,898 | 69,394 / 67,618 |
| CLIP đọc toàn sketch, hạng tử ngữ nghĩa gốc | 82,857 / 83,236 | 75,519 / 74,352 |

So với bối cảnh che vùng, log score của bản vẽ độc lập đổi **−0,00061 / +0,00285 nat**. Cả hai khoảng 95% khi bootstrap lớp đều chứa 0. Log score của ảnh thật cùng nguồn tăng **0,03034 / 0,00198 nat**. Lợi ích riêng cho bản vẽ cùng nguồn, sau khi trừ thay đổi tương ứng ở nguồn khác cùng lớp, là **+0,00299 / −0,00293 nat**. Chưa có bằng chứng nhất quán cho việc phục hồi một ý nghĩa đặc thù của ảnh gốc. 64 từ hàng đầu của bối cảnh che vùng bao phủ **23,96% / 20,91%** khối lượng từ của bản vẽ khác. Cả đối tượng đo lẫn bối cảnh đều khác pilot góc A, nên hai độ phủ không phải ablation một biến.

Mẫu 768 truy vấn mỗi split loại mọi ID từ pilot trước ở cả hai split. Cấu hình được chấm riêng trên gallery đầy đủ, rồi đánh giá các dự đoán đã khóa. Phép thử này chưa đạt điều kiện chuyển sang TU hoặc thay GCWE của baseline. [Dẫn xuất và thiết kế khóa](protocol.md) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [Kết quả theo truy vấn](paired-T4.zip)
