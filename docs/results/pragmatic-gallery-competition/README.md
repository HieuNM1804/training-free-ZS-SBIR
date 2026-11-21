# Kiểm chứng từ phân biệt được hai nhóm gallery cạnh tranh

Ở mỗi nhánh của cây gallery, tính khả năng một từ phân biệt hai nhóm, kể cả khi bỏ ảnh cho bằng chứng mạnh nhất ở một nhóm và bỏ ảnh yếu nhất ở nhóm còn lại. Dùng xác suất đó để xây mô tả cho từng đường đi.

| mAP / precision (%) | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin 30 lớp |
|---|---:|---:|---:|
| Baseline từ và diffusion | 81,86 / 81,33 | 74,19 / 72,68 | 79,16 / 80,24 |
| Mô tả có cận bỏ một ảnh | 80,70 / 81,35 | 73,93 / 72,39 | 77,81 / 80,00 |
| Gộp nhóm tối ưu riêng | 81,86 / 81,33 | 74,19 / 72,68 | 79,16 / 80,24 |
| Bỏ kiểm tra độ bền của từ | 80,77 / 81,35 | 73,94 / 72,41 | 77,85 / 80,01 |

Biến thể chính thua baseline về mAP cả ba lần. Cận bỏ một ảnh đúng về mặt số học trên các trường hợp đã thử. Đối chứng gộp tuyến tính xác nhận xác suất các nhánh triệt tiêu theo lý thuyết khi chưa can thiệp. Độ bền của một từ khi bỏ ảnh chưa đủ để cho rằng từ đó giúp nhận dạng sketch tốt hơn.

CLIP ViT-B/32 đóng băng. Mỗi cấu hình cuối được chấm bằng hàm KD-SBIR gốc trên 14.179, 12.694 và 2.130 sketch ở Colab L4. Ext-1 và TU dùng mAP@all và P@100. Ext-2 dùng mAP@200 và P@200. Chọn trên hai split Sketchy unseen rồi khóa cấu hình trước khi chấm TU. Ở cả hai benchmark, chiều truy hồi là sketch → photo. [Giao thức](../../PRAGMATIC_GALLERY_PROTOCOL.md) · [Khóa](freeze.json) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU](tuberlin.json) · [Từng truy vấn](per-query.zip)
