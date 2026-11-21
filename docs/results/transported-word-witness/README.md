# Chuyển nghĩa thị giác của từ giữa hai prompt theo gallery

Ước lượng tâm thị giác cho từng tên từ hai nửa gallery. Dùng phép quay góc nhỏ nhất giữa cặp prompt và độ lặp lại của tâm để điều chỉnh cách đọc từ ở sketch. Mỗi ảnh lấy trách nhiệm thành phần từ nửa gallery đối diện.

| mAP / precision (%) | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin 30 lớp |
|---|---:|---:|---:|
| Baseline từ và diffusion | 81,86 / 81,33 | 74,19 / 72,68 | 79,16 / 80,24 |
| Phép quay có kiểm tra qua hai nửa gallery | 78,42 / 78,47 | 73,12 / 71,45 | 74,04 / 75,66 |
| Tối ưu riêng, giữ nguyên hướng từ của sketch | 81,31 / 81,66 | 75,07 / 73,55 | 76,39 / 77,90 |
| Dịch chuyển Euclid | 78,47 / 78,52 | 73,25 / 71,54 | 74,05 / 75,55 |
| Hoán vị hướng điều chỉnh | 78,95 / 79,56 | 69,54 / 67,17 | 72,02 / 72,68 |

Phép quay tái tạo đúng cặp hướng prompt và bảo toàn góc theo phép kiểm tra. mAP của biến thể chính giảm mạnh nhất trên TU. Tâm thị giác có thể tái lập giữa các phần của cùng gallery mà vẫn cung cấp hướng bất lợi cho miền sketch. Đợt này không ủng hộ việc dùng tâm ấy để dịch chuyển cách mô hình hiểu từ.

CLIP ViT-B/32 đóng băng. Mỗi cấu hình cuối được chấm bằng hàm KD-SBIR gốc trên 14.179, 12.694 và 2.130 sketch ở Colab L4. Ext-1 và TU dùng mAP@all và P@100. Ext-2 dùng mAP@200 và P@200. Chọn trên hai split Sketchy unseen rồi khóa cấu hình trước khi chấm TU. Ở cả hai benchmark, chiều truy hồi là sketch → photo. [Giao thức](../../TRANSPORTED_WORD_PROTOCOL.md) · [Khóa](freeze.json) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU](tuberlin.json) · [Từng truy vấn](per-query.zip)
