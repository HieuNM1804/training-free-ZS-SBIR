# Đối chiếu bố cục của sketch và photo qua từng từ

Giữ nguyên các patch của một ảnh và đổi vị trí của chúng trước khi cộng embedding vị trí. Với mỗi từ công khai, đo điểm CLIP thay đổi bao nhiêu. Phương pháp dùng thay đổi ở cả sketch và photo để điều chỉnh trọng số khớp từ.

| mAP / precision (%) | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin 30 lớp |
|---|---:|---:|---:|
| Ghép bằng chứng bố cục theo từ | 83,17 / 81,99 | 75,15 / 73,56 | 77,88 / 77,66 |
| Đối chứng tương phản được tối ưu trên Sketchy | 83,40 / 82,08 | 75,34 / 73,78 | 78,86 / 78,30 |
| Cùng prior, dùng ảnh nguyên vẹn | 83,25 / 81,98 | 74,89 / 73,53 | 78,09 / 77,69 |
| Khớp từ và diffusion RGB | 81,86 / 81,33 | 74,19 / 72,68 | 79,16 / 80,24 |

So với cùng prior, ghép bố cục theo từ thay đổi mAP −0,08 trên Ext-1, +0,26 trên Ext-2 và −0,21 trên TU. Đối chứng tương phản đơn giản đạt điểm cao hơn cấu hình ghép trên cả ba split. Phiên chạy lại dùng cùng cấu hình bố cục từng được hai split Sketchy chọn. Tích lũy thống kê theo phần nhỏ đã cho phép chấm toàn bộ 27.262 ảnh TU trên L4.

Chọn cấu hình trên hai split Sketchy unseen, dùng trung bình mAP rồi đến precision khi hòa. TU-Berlin dùng cấu hình đã khóa. Cả hai benchmark đều đo chiều sketch → photo. OpenAI CLIP ViT-B/32 được đóng băng. Hàm KD-SBIR gốc chấm trên đủ 14.179, 12.694 và 2.130 sketch cho mỗi cấu hình cuối. Ext-1 và TU dùng mAP@all và P@100. Ext-2 dùng mAP@200 và P@200. [Giao thức](../../LAYOUT_WORD_PROTOCOL.md) · [Khóa](freeze.json) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU](tuberlin.json) · [Từng truy vấn](per-query.zip)
