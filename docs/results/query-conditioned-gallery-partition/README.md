# Chọn mức gộp ảnh theo từng sketch

Mỗi ô là mAP / precision theo %. Sketchy Ext-1 dùng mAP@all / P@100, Ext-2 dùng mAP@200 / P@200, TU-Berlin dùng mAP@all / P@100. Ba split có lần lượt 14.179, 12.694 và 2.130 truy vấn, trên gallery 17.100, 12.552 và 27.262 ảnh. Cấu hình được chọn trên hai split unseen Sketchy với trọng số mAP bằng nhau, rồi khóa trước khi chấm 30 lớp TU-Berlin. Tất cả dùng Colab L4 và CLIP ViT-B/32 đóng băng. [Commit khóa](https://github.com/HieuNM1804/training-free-ZS-SBIR/commit/0c82ac0965cec6e52ced09c5c8bc9438efd006ed) · [Mô hình và ba nguồn trực tiếp](../../QUERY_CONDITIONED_GALLERY_PARTITION.md)

| Phép đo | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin |
|---|---:|---:|---:|
| Khớp trực tiếp 20.931 từ | 75,31 / 78,94 | 71,42 / 70,28 | 73,35 / 77,22 |
| Dùng cosine sketch chọn mức gộp của cây ảnh | 80,44 / 80,87 | 72,82 / 71,52 | 77,14 / 78,77 |
| Chọn một mức cây cố định | 80,53 / 81,47 | 73,03 / 72,07 | 77,07 / 78,97 |
| Dùng chính điểm từ để chọn mức gộp | 80,43 / 81,07 | 73,16 / 72,20 | 76,84 / 78,86 |
| Trộn các mức theo prior cố định, tốt nhất trong nhánh từ trực tiếp | 80,97 / 81,49 | 73,64 / 72,43 | 78,17 / 79,52 |
| Nền từ, vùng ảnh và diffusion | 84,37 / 83,85 | 77,01 / 74,84 | 80,07 / 80,92 |
| Chọn mức theo sketch trên nền từ/vùng | 84,96 / 83,95 | 77,05 / 74,95 | 80,03 / 80,93 |
| Mức gộp cố định trên nền từ/vùng | 85,11 / 84,50 | 77,45 / 75,46 | 80,02 / 80,79 |

Việc chọn các ảnh giúp nhau chấm từ trong cùng nhóm làm mAP tăng so với khớp từ trực tiếp ở cả ba split. Tuy nhiên, phần đóng góp của việc tự tìm mức phân chia cho mỗi sketch cho chênh lệch so với mức cố định là −0,092 / −0,210 / +0,072 điểm. Các khoảng bootstrap 95% theo lớp đều chứa 0, trên TU từ −0,95 đến +1,47 điểm. Trên hệ thống vùng/từ mạnh, cấu hình điểm cao nhất trên hai split Sketchy là đối chứng mức cố định. Đây là kết quả có ý nghĩa với lựa chọn cơ chế. Trong repo đã có đối chứng gộp ảnh theo FINCH từ trước trong `tf_sbir/cluster_prior.py`.

Hai ảnh giữ lại dùng làm ứng viên không trao đổi điểm với sketch khác. Ba phần của gallery tách riêng việc ước lượng độ đồng nhất, đo bằng chứng và chấm ứng viên. Lưới pilot gồm 1.265 hàng. Bước toàn Sketchy còn 180 hàng, rồi phát lại tất cả thứ hạng của 42 hàng cuối trên cả ba split bằng hàm KD-SBIR gốc. Sai số lớn nhất giữa metric vector hóa và hàm gốc là 2,39 × 10⁻⁷. Các file ZIP chứa số liệu theo truy vấn, kèm SHA-256 trong JSON cùng tên.
