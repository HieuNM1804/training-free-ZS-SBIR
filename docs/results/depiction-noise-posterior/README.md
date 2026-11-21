# Ước lượng lại xác suất từ theo độ biến thiên của từng bản vẽ

Các chiều từ nhạy với thay đổi vùng nhìn được ước lượng từ ảnh gallery không nhãn. Từng sketch tự cung cấp độ biến thiên theo các chiều đó. Một mô hình Gaussian ước lượng phân bố hậu nghiệm của bằng chứng từ, rồi chấm cùng các photo như đối chứng.

| mAP / precision (%) | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin 30 lớp |
|---|---:|---:|---:|
| Hậu nghiệm của từng sketch | 83,49 / 83,09 | 76,12 / 73,96 | 79,20 / 79,80 |
| Dùng phương sai chung của gallery | 83,20 / 82,92 | 75,72 / 73,54 | 78,99 / 79,37 |
| Giữ nguyên bằng chứng từ các vùng phóng to | 84,00 / 83,45 | 76,43 / 74,17 | 79,50 / 80,13 |
| Khớp từ + diffusion RGB | 81,86 / 81,33 | 74,19 / 72,68 | 79,16 / 80,24 |

Phương sai theo từng sketch cải thiện mAP so với phương sai chung lần lượt +0,29, +0,40 và +0,22 điểm. So với giữ nguyên bằng chứng vùng, mức chênh là −0,51, −0,31 và −0,30. Tín hiệu biến thiên dự báo được sai khác giữa hai người vẽ, nhưng thu hẹp toàn bộ vector bằng chứng theo mô hình này lại làm truy hồi kém đi.

Chọn cấu hình trên hai split Sketchy unseen, dùng trung bình mAP rồi đến precision khi hòa. TU-Berlin dùng cấu hình đã khóa. Cả hai benchmark đều đo chiều sketch → photo. OpenAI CLIP ViT-B/32 được đóng băng. Hàm KD-SBIR gốc chấm trên đủ 14.179, 12.694 và 2.130 sketch cho mỗi cấu hình cuối. Ext-1 và TU dùng mAP@all và P@100. Ext-2 dùng mAP@200 và P@200. [Giao thức](../../DEPICTION_POSTERIOR_PROTOCOL.md) · [Khóa](freeze.json) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU](tuberlin.json) · [Từng truy vấn](per-query.zip)
