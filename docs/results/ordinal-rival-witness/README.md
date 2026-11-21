# So sánh các cặp từ giữa các nhóm ảnh cạnh tranh

Mỗi cặp nhóm ảnh gallery đề cử những cặp từ công khai có thứ tự xuất hiện lặp lại trên hai nửa gallery. Từng sketch trả lời các so sánh này. Hệ thống kết hợp chúng với điểm khớp từ và diffusion RGB.

| mAP / precision (%) | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin 30 lớp |
|---|---:|---:|---:|
| So sánh cặp từ đã khóa | 83,26 / 82,28 | 74,35 / 71,87 | 77,32 / 79,90 |
| Gộp các cạnh với trọng số đều | 82,10 / 81,35 | 73,34 / 71,54 | 77,47 / 79,75 |
| Đối chứng dùng độ chênh liên tục | 82,20 / 81,48 | 73,91 / 72,15 | 77,84 / 79,84 |
| Khớp từ và diffusion RGB | 81,86 / 81,33 | 74,19 / 72,68 | 79,16 / 80,24 |

So với phương pháp gốc, cấu hình đã khóa sửa được 692 lỗi top-1 trên Ext-2 và tạo thêm 381 lỗi. P@200 giảm 0,81 điểm trên split này. mAP trên TU giảm 1,85 điểm. Điểm Ext-1 cao hơn đi cùng hiệu ứng chuyển giao yếu.

Chọn cấu hình trên hai split Sketchy unseen, dùng trung bình mAP rồi đến precision khi hòa. TU-Berlin dùng cấu hình đã khóa. Cả hai benchmark đều đo chiều sketch → photo. OpenAI CLIP ViT-B/32 được đóng băng. Hàm KD-SBIR gốc chấm trên đủ 14.179, 12.694 và 2.130 sketch cho mỗi cấu hình cuối. Ext-1 và TU dùng mAP@all và P@100. Ext-2 dùng mAP@200 và P@200. [Giao thức](../../ORDINAL_RIVAL_PROTOCOL.md) · [Khóa](freeze.json) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU](tuberlin.json) · [Từng truy vấn](per-query.zip)
