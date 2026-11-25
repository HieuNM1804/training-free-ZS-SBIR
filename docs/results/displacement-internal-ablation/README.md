# Ablation bên trong cấu hình displacement-consensus

Colab Tesla T4, seed 42, OpenAI CLIP ViT-B/32 đóng băng, từng sketch độc lập, từ điển Open Images, không Hierarchical. Tất cả biến thể dùng cùng gallery, evaluator KD-SBIR gốc và hệ số điểm đã khóa. Lượt chấm `e8319da` gồm 14.179 truy vấn Ext-1, 12.694 truy vấn Ext-2 và 2.130 truy vấn TU-Berlin đủ 30 lớp. Khi phát lại hệ đầy đủ, sai lệch tuyệt đối trung bình theo truy vấn so với bảng chính trước đó dưới 0,0003 điểm phần trăm trên cả ba split. [Protocol](protocol.md)

Các ô ghi mAP@all / P@100 trên Ext-1 và TU-Berlin, mAP@200 / P@200 trên Ext-2. Đơn vị là phần trăm.

| Biến thể | Sketchy Ext‑1 | Sketchy Ext‑2 | TU‑Berlin 30 lớp |
|---|---:|---:|---:|
| Đầy đủ | 85,09 / 84,60 | 78,41 / 76,17 | 81,87 / 82,98 |
| Tắt RGB diffusion ở cả ba vị trí | 80,83 / 83,80 | 76,53 / 74,87 | 78,16 / 80,92 |
| Tắt RGB diffusion ở ngữ nghĩa toàn ảnh (S) | 82,42 / 84,01 | 76,99 / 75,39 | 79,46 / 81,40 |
| Tắt RGB diffusion ở ngân hàng từ vùng (R) | 84,08 / 84,51 | 77,56 / 75,64 | 81,03 / 82,59 |
| Tắt RGB diffusion trước và sau cập nhật từ của GCWE (G) | 84,08 / 84,68 | 77,89 / 75,98 | 80,93 / 82,61 |
| Bỏ chọn vùng theo từng ứng viên | 84,76 / 84,32 | 77,99 / 75,80 | 81,69 / 82,86 |
| Bỏ hiệu chỉnh độ phổ biến của vùng theo gallery | 84,98 / 84,52 | 78,26 / 76,02 | 81,69 / 82,81 |
| Bỏ prototype ảnh trong ngân hàng từ vùng | 85,18 / 84,55 | 78,07 / 76,00 | 81,53 / 82,41 |
| Bỏ trọng số prior của từ vùng, giữ nguyên 64 từ | 85,18 / 84,73 | 78,13 / 75,96 | 81,84 / 83,04 |
| Bỏ lật ngang ở S và R | 84,60 / 84,13 | 77,90 / 75,69 | 81,26 / 82,31 |
| Chỉ dùng 4 vùng lớn | 85,01 / 84,49 | 78,43 / 76,16 | 81,94 / 83,05 |
| Chỉ dùng 9 vùng nhỏ | 83,33 / 83,04 | 76,95 / 74,71 | 80,37 / 81,51 |
| Bỏ đối chứng canvas đồng màu trong GCWE | 85,06 / 84,57 | 78,23 / 76,01 | 81,88 / 82,58 |
| Bỏ pseudocount của donor trong GCWE | 85,02 / 84,54 | 78,31 / 76,08 | 81,86 / 82,99 |
| Bỏ hiệu chỉnh bằng donor đồng đều trong GCWE | 81,74 / 80,65 | 75,35 / 73,54 | 77,32 / 77,30 |
| Chỉ dùng donor có điểm hình học cao nhất | 79,41 / 77,78 | 73,15 / 70,37 | 78,24 / 78,19 |
| Dùng cùng tập donor cho cả gallery, bỏ cross-fitting | 85,08 / 84,57 | 78,41 / 76,20 | 81,90 / 82,95 |

## Kết quả

- Tắt toàn bộ RGB diffusion giảm 4,26 / 1,88 / 3,71 điểm mAP. Tắt riêng ở nhánh ngữ nghĩa toàn ảnh giảm 2,67 / 1,42 / 2,41 điểm. Tắt riêng ở ngân hàng từ vùng giảm 1,00 / 0,85 / 0,84 điểm. Tắt riêng ở GCWE giảm 1,01 / 0,52 / 0,94 điểm. Các thay đổi về mAP được tính từ thứ hạng nên có thể tương tác khi tắt đồng thời.
- Chỉ dùng donor hình học đứng đầu giảm 5,68 / 5,25 / 3,62 điểm mAP. Bỏ hiệu chỉnh theo bằng chứng donor đồng đều giảm 3,34 / 3,06 / 4,55 điểm. Khoảng bootstrap ghép cặp theo lớp của phép thử donor đứng đầu nằm hoàn toàn trên 0 trên cả ba split khi tính điểm đầy đủ trừ điểm ablation. Với hiệu chỉnh donor đồng đều, khoảng Ext-2 bao gồm 0.
- Chọn vùng riêng cho từng ứng viên tăng 0,32 / 0,41 / 0,18 điểm mAP so với trung bình theo prior vùng cố định. Khoảng bootstrap TU bao gồm 0. Bỏ 9 vùng nhỏ và giữ 4 vùng lớn làm mAP thay đổi −0,07 / +0,02 / +0,07 điểm và precision thay đổi −0,11 / −0,01 / +0,07 điểm. Hướng lật ngang trong S và R đem lại 0,49 / 0,50 / 0,61 điểm mAP.
- Bỏ cross-fitting, bỏ pseudocount và bỏ đối chứng canvas đồng màu thay đổi ít về mAP. Canvas vẫn làm tăng 0,40 điểm P@100 trên TU. Cross-fitting giữ donor và ảnh đích khác nửa ở bước cập nhật từ, trước khi diffusion trên toàn gallery. Các phép thử được chấm riêng nên chưa có số cho cấu hình bỏ đồng thời các bước này.

## Các đối chứng đã có trên cùng cấu hình

| Biến thể | Sketchy Ext‑1 | Sketchy Ext‑2 | TU‑Berlin 30 lớp |
|---|---:|---:|---:|
| Xáo tọa độ patch ảnh | 84,94 / 84,40 | 78,02 / 75,69 | 81,55 / 82,13 |
| Bỏ bước gom cặp patch theo độ dịch chuyển | 84,94 / 84,35 | 78,11 / 75,73 | 81,73 / 82,45 |
| Bỏ trừ phân bố dịch chuyển kỳ vọng | 84,97 / 84,40 | 78,51 / 76,21 | 81,45 / 82,45 |
| Thay thống kê dịch chuyển bằng GCWE gốc | 84,78 / 84,25 | 77,86 / 75,67 | 81,49 / 82,18 |

[Đối chứng hình học đã lưu](../gcwe-displacement-full/README.md) · [Factorial S/R/G đã lưu](../three-component-factorial/) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU-Berlin](tuberlin.json) · [Kết quả từng truy vấn](paired-T4.zip)
