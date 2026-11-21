# Ước lượng xác suất sketch thể hiện một phần các khái niệm của ảnh

Ảnh được biểu diễn bằng xác suất của 64 từ công khai do gallery chọn. Với từng sketch, phương pháp tính chính xác kỳ vọng của từ có điểm cao nhất trong hai lần lấy mẫu từ phân bố của ảnh. Hai lượt lấy mẫu tạo mô hình cho việc sketch chỉ thể hiện một phần nội dung.

| mAP / precision (%) | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin 30 lớp |
|---|---:|---:|---:|
| Chọn từ theo sketch trong hai lượt | 82,95 / 82,12 | 75,74 / 73,66 | 78,55 / 79,02 |
| Một lượt, cấu hình tuyến tính tối ưu | 82,86 / 82,18 | 75,95 / 73,85 | 78,32 / 78,86 |
| Ưu tiên từ theo độ phổ biến trong gallery | 82,61 / 82,25 | 76,40 / 74,19 | 77,77 / 78,74 |
| Khớp từ và diffusion RGB | 81,86 / 81,33 | 74,19 / 72,68 | 79,16 / 80,24 |

So với một lượt, chọn từ qua hai lượt làm mAP thay đổi +0,09, −0,22 và +0,23 điểm. Đối chứng dùng độ phổ biến của từ đạt 76,40 mAP trên Ext-2. Các số liệu ủng hộ việc rút gọn và gán từ theo gallery trên Sketchy, trong khi lợi ích riêng của lựa chọn theo từng sketch còn yếu.

Chọn cấu hình trên hai split Sketchy unseen, dùng trung bình mAP rồi đến precision khi hòa. TU-Berlin dùng cấu hình đã khóa. Cả hai benchmark đều đo chiều sketch → photo. OpenAI CLIP ViT-B/32 được đóng băng. Hàm KD-SBIR gốc chấm trên đủ 14.179, 12.694 và 2.130 sketch cho mỗi cấu hình cuối. Ext-1 và TU dùng mAP@all và P@100. Ext-2 dùng mAP@200 và P@200. [Giao thức](../../SELECTIVE_LISTENER_PROTOCOL.md) · [Khóa](freeze.json) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU](tuberlin.json) · [Từng truy vấn](per-query.zip)
