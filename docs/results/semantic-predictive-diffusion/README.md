# Hiệu ứng của suy luận ngữ nghĩa khi giữ nguyên đồ thị diffusion

Đồ thị RGB và toán tử khuếch tán đã khóa được dùng cho mọi cách đọc. Đối chứng hiện có lấy kỳ vọng của bằng chứng từ. Phương pháp mới lan truyền log likelihood cho từng giả thuyết về tỷ lệ ảnh ủng hộ sketch, rồi lấy tích phân trên các giả thuyết ấy. Nghiên cứu thăm dò dùng 1.536 sketch mỗi split, toàn bộ gallery và hai metric KD-SBIR.

| Cách đọc, mAP / precision (%) | Ext-1 | Ext-2 |
|---|---:|---:|
| Diffusion thông thường | 81,84 / 80,96 | 73,79 / 72,31 |
| Tỷ lệ 1/2 | 81,36 / 81,39 | 74,28 / 73,01 |
| Toàn bộ nhóm mang bằng chứng | 80,14 / 81,12 | 74,35 / 73,12 |
| Suy luận tỷ lệ chưa biết, cỡ nhóm 16 | 81,25 / 81,36 | 74,29 / 73,11 |

So với diffusion thông thường, phương pháp chính làm mAP thay đổi −0,59 / +0,50 điểm ở Ext-1 / Ext-2. Giảm cỡ nhóm xuống 4 hoặc 8 giữ nguyên chiều của hiệu ứng. Trên cùng đồ thị, bằng chứng về tính nhất quán trong nhóm giúp precision ở cả hai split nhưng chưa làm mAP cao hơn trên cả hai. Không chọn một cấu hình theo riêng Ext-2. Gấp đôi số nút tích phân ở batch kiểm tra làm mAP chênh dưới 0,00001 điểm trên cả hai split.

Mã `4be9f67`, Colab T4, frozen CLIP ViT-B/32, seed 42. [Mô hình và đối chứng](protocol.md) · [Ext-1](sketchy_1-pilot.json) · [Ext-2](sketchy_2-pilot.json) · [Bản ghi theo truy vấn](paired-T4.zip)
