# Dự báo độ ổn định của thứ hạng khi cùng vật thể được vẽ lại

Các vùng của một sketch cung cấp nhiều lần đọc cùng các nhóm ảnh gallery. Phương pháp ước lượng phương sai của từng hiệu điểm, rồi chuyển xác suất thắng từng đối thủ trở về thang điểm truy hồi. Dữ liệu các bản vẽ cùng vật thể chỉ tham gia phép kiểm chứng hồi cứu.

| mAP / precision (%) | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin 30 lớp |
|---|---:|---:|---:|
| Dùng covariance của các điểm trên cùng sketch | 83,96 / 83,44 | 76,40 / 74,16 | 79,52 / 80,14 |
| Bỏ covariance giữa hai nhóm ảnh | 83,97 / 83,43 | 76,39 / 74,13 | 79,36 / 80,15 |
| Dùng covariance cố định từ gallery | 83,98 / 83,49 | 76,39 / 74,13 | 79,59 / 80,16 |
| Gộp các vùng phóng to của sketch | 84,00 / 83,45 | 76,43 / 74,17 | 79,50 / 80,13 |

AUROC dự báo bản vẽ khác có giữ cùng thứ tự tăng từ 0,723 lên 0,759 trên Ext-1 và từ 0,690 lên 0,759 trên Ext-2. Với truy hồi thực tế, chênh mAP giữa phương pháp và điểm đầu vào lần lượt là −0,03, −0,03 và +0,02. Độ ổn định của một lựa chọn và việc lựa chọn đó khớp lớp đúng là hai đại lượng khác nhau. [Kiểm chứng thứ hạng của các bản vẽ cùng vật thể](sibling-margin-diagnosis.json)

Chọn cấu hình trên hai split Sketchy unseen, dùng trung bình mAP rồi đến precision khi hòa. TU-Berlin dùng cấu hình đã khóa. Cả hai benchmark đều đo chiều sketch → photo. OpenAI CLIP ViT-B/32 được đóng băng. Hàm KD-SBIR gốc chấm trên đủ 14.179, 12.694 và 2.130 sketch cho mỗi cấu hình cuối. Ext-1 và TU dùng mAP@all và P@100. Ext-2 dùng mAP@200 và P@200. [Giao thức](../../REDRAW_RANK_PROTOCOL.md) · [Khóa](freeze.json) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU](tuberlin.json) · [Từng truy vấn](per-query.zip)
