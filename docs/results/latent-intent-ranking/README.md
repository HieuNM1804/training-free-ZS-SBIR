# Chọn thứ hạng theo AP kỳ vọng của từng cách hiểu truy vấn

Mỗi sketch đề xuất một số cách hiểu từ từ điển công khai. Mô hình ước lượng xác suất liên quan đồng thời của từng cặp nhóm ảnh trong gallery, rồi dùng đại lượng này để chọn giữa thứ hạng nền và các phương án chuyển một phần trọng số sang một từ. Cấu hình khóa dùng 16 tên, giữ khối lượng 0,85 cho giả thuyết của phương pháp nền và cho phép mức pha tối đa 0,12.

| mAP / precision (%) | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin 30 lớp |
|---|---:|---:|---:|
| Điểm nền theo vùng | 84,37 / 83,85 | 77,01 / 74,84 | 80,07 / 80,92 |
| AP kỳ vọng với các cách hiểu riêng | 84,50 / 83,94 | 76,82 / 74,89 | 80,40 / 81,17 |
| Gộp các cách hiểu trước khi tính đồng liên quan | 84,46 / 83,92 | 76,70 / 74,83 | 80,48 / 81,29 |
| Điểm kỳ vọng bậc một | 84,44 / 83,90 | 76,35 / 74,71 | 80,47 / 81,24 |
| Chỉ dùng giả thuyết của thứ hạng nền | 84,48 / 83,93 | 76,87 / 74,95 | 80,36 / 81,16 |
| Giữ quyết định đổi, thay từ đã chọn bằng từ đầu tiên | 84,28 / 83,85 | 76,66 / 74,68 | 80,03 / 80,67 |
| Đối chứng bậc một đã được dò hệ số riêng | 84,55 / 83,97 | 76,39 / 74,85 | 80,52 / 81,31 |

Trên Ext-1, chênh lệch mAP ghép cặp so với nền là +0,136 điểm với khoảng bootstrap 95% theo lớp [+0,079, +0,203]. Trên Ext-2, mức thay đổi là −0,190 [−0,407, +0,008]. Trên TU-Berlin, mức thay đổi là +0,326 [+0,110, +0,623]. Riêng tại Ext-2, 153 lỗi ở vị trí đầu được sửa và 72 truy vấn mất kết quả đúng ở vị trí đầu. Trong 5.303 truy vấn có đổi thứ hạng, AP thực tế trung bình giảm 0,456 điểm. Mục tiêu xấp xỉ AP trên toàn gallery có thể bất đồng với AP@200 thực tế. Đối chứng bậc một được tối ưu riêng đạt 80,52 mAP trên TU. Bằng chứng hiện tại chưa đủ quy mức tăng trên TU cho việc mô hình hóa đồng liên quan.

Chọn cấu hình trên hai split Sketchy unseen, dùng trung bình mAP rồi đến precision khi hòa. TU-Berlin dùng cấu hình đã khóa. Cả hai benchmark đều đo chiều sketch → photo. OpenAI CLIP ViT-B/32 được đóng băng. Hàm KD-SBIR gốc chấm trên đủ 14.179, 12.694 và 2.130 sketch cho mỗi cấu hình cuối. Ext-1 và TU dùng mAP@all và P@100. Ext-2 dùng mAP@200 và P@200. Mẫu thăm dò cân bằng và phép dò trên toàn bộ truy vấn dùng trọng số lớp khác nhau. Cấu hình được chốt lại trên hai split đầy đủ. [Kết quả mẫu](pilot.json) · [Quy trình lần đo đầy đủ](full-protocol.json) [Giao thức](../../LATENT_INTENT_FULL_LOCK.md) · [Khóa](freeze.json) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU](tuberlin.json) · [Từng truy vấn](per-query.zip)
