# S/R/G tối giản với patch CLIP

Colab Tesla T4, CLIP ViT-B/32 đóng băng, seed 42 và evaluator KD-SBIR gốc. Mã chạy ở commit `71892c8`. Chấm đủ unseen Ext-1 (14.179 sketch), Ext-2 (12.694 sketch) và TU-Berlin (2.130 sketch, 30 lớp). Giữ hệ số 0,65 cho S và G, 0,35 cho R. Xem [protocol](protocol.md) để biết chính xác các phần giữ lại và bỏ đi.

| Cấu hình | Ext-1 mAP@all / P@100 | Ext-2 mAP@200 / P@200 | TU mAP@all / P@100 |
|---|---:|---:|---:|
| S + R + G tối giản | 76,74 / 80,68 | 73,20 / 71,64 | 75,87 / 79,92 |
| S + G | 76,85 / 80,79 | 73,52 / 71,90 | 75,97 / 79,91 |
| S + R | 76,13 / 79,95 | 72,15 / 71,12 | 73,98 / 78,29 |
| S | 76,22 / 80,02 | 72,45 / 71,33 | 74,13 / 78,28 |
| Bản cũ chỉ tắt diffusion | 80,83 / 83,80 | 76,53 / 74,87 | 78,16 / 80,92 |
| Bản cũ đầy đủ | 85,09 / 84,60 | 78,41 / 76,17 | 81,87 / 82,98 |

Thêm R với trung bình đều 49 patch vào S+G làm mAP giảm 0,10 / 0,32 / 0,10 điểm. Khoảng bootstrap theo lớp chứa 0 ở cả ba split. So với bản cũ chỉ tắt diffusion, bản tối giản giảm 4,08 / 3,32 / 2,28 điểm mAP. Phép so này đổi toàn bộ cách đọc vùng, gồm cách tạo ngân hàng từ vùng, crop và trọng số vùng.

Thêm G vào S làm mAP tăng 0,63 / 1,07 / 1,84 điểm. Khoảng bootstrap theo lớp 95% lần lượt là [−0,21, +1,66], [−0,87, +3,29] và [+0,65, +3,08]. Đối chiếu ID truy vấn, gallery và nhãn giữa lượt mới và baseline đã lưu. Sai lệch tuyệt đối trung bình của phép chạy lại S so với S trực tiếp đã lưu dưới 0,005 điểm trên cả hai metric.

[Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU-Berlin](tuberlin.json) · [Số liệu từng truy vấn](paired-T4.zip) · [Bản cũ tắt diffusion và bản đầy đủ](../displacement-internal-ablation/README.md)
