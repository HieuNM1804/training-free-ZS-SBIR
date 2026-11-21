# Kiểm tra bằng chứng của cùng một vùng khi giữ và bỏ patch

Với mỗi vị trí ở hai kích thước cửa sổ, CLIP mã hóa phần được giữ, phần còn lại và ảnh phóng to của cửa sổ. Các phản hồi theo từng từ dùng để chọn vị trí lấy bằng chứng cho truy hồi. Đối chứng phóng to dùng cùng các cửa sổ và cùng chỉ mục ảnh.

| mAP / precision (%) | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin 30 lớp |
|---|---:|---:|---:|
| Giữ và bỏ vùng tương ứng | 83,99 / 83,38 | 76,63 / 74,45 | 79,18 / 79,58 |
| Chỉ phóng to và gộp các cửa sổ | 84,00 / 83,45 | 76,43 / 74,17 | 79,50 / 80,13 |
| Cùng chỉ mục, dùng sketch gốc | 82,86 / 82,18 | 75,95 / 73,85 | 78,32 / 78,86 |
| Khớp từ + diffusion RGB | 81,86 / 81,33 | 74,19 / 72,68 | 79,16 / 80,24 |

So với phóng to đơn thuần, chênh lệch mAP lần lượt là −0,007, +0,205 và −0,323 điểm. Cả ba khoảng tin cậy 95% bootstrap theo lớp đều chứa 0. Với cùng chỉ mục ảnh, phóng to giúp tăng mAP trên cả ba split. Nếu dùng nhãn để chọn hồi cứu cửa sổ có AP tốt nhất cho mỗi truy vấn, mức tăng tối đa trên mẫu pilot lớn hơn nhiều. Đó là phép chẩn đoán dư địa chọn vùng. [Chẩn đoán chọn vùng trên Sketchy](view-diagnosis.json)

Chọn cấu hình trên hai split Sketchy unseen, dùng trung bình mAP rồi đến precision khi hòa. TU-Berlin dùng cấu hình đã khóa. Cả hai benchmark đều đo chiều sketch → photo. OpenAI CLIP ViT-B/32 được đóng băng. Hàm KD-SBIR gốc chấm trên đủ 14.179, 12.694 và 2.130 sketch cho mỗi cấu hình cuối. Ext-1 và TU dùng mAP@all và P@100. Ext-2 dùng mAP@200 và P@200. [Giao thức](../../REGIONAL_EVIDENCE_PROTOCOL.md) · [Khóa](freeze.json) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU](tuberlin.json) · [Từng truy vấn](per-query.zip)
