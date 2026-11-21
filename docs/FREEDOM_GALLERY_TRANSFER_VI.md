# Chuyển FreeDom từ composed image retrieval sang SBIR

Với bộ nhớ giới hạn ở gallery ảnh không nhãn, đợt này tái hiện hai bước của FreeDom. Mỗi truy vấn gồm một sketch và 19 ảnh proxy gần nhất bằng cosine RGB. Mỗi thành viên bỏ phiếu cho bảy từ nguyên văn trong Open Images. Giữ bảy từ đạt nhiều phiếu nhất, ghép từng từ với hậu tố `photo`, mã hóa CLIP và tính tổng có trọng số theo tần suất. Thứ tự hòa phiếu theo thứ tự từ được gặp. Do mỗi query sketch nằm ngoài gallery, luôn thêm chính sketch vào 20 thành viên. Các prompt và cách ghép này khác lượt chuyển FreeDom trước đây của repo. Code tác giả dùng backbone lớn hơn cho thí nghiệm CLIP. Lượt SBIR dùng ViT-B/32 đóng băng cho mọi phương pháp. [FreeDom, WACV 2025](https://arxiv.org/abs/2412.03297) · [Mã tham chiếu của tác giả](https://github.com/NikosEfth/freedom/blob/main/utils_retrieval.py)

| Cấu hình | Ext-1 mAP@all | Ext-2 mAP@200 | TU mAP@all |
|---|---:|---:|---:|
| FreeDom, 20 thành viên với proxy tìm bằng RGB | 45,18 | 40,36 | 53,83 |
| Chỉ sketch tự chọn từ | 63,23 | 61,60 | 56,66 |
| 96 thành viên với proxy tìm bằng RGB | 52,46 | 47,27 | 52,93 |
| 96 thành viên với proxy tìm qua khái niệm | **77,99** | **72,30** | **73,91** |

Giữ số thành viên ở 96 và cùng phép bỏ phiếu, việc thay cách tìm proxy cho mức tăng mAP trên cả ba split. Trong phép chẩn đoán trên mẫu cùng 19 proxy, độ thuần lớp tăng từ 38,63% lên 77,99% trên Ext-1, từ 33,15% lên 69,64% trên Ext-2. Việc đếm ảnh đúng lớp chỉ thực hiện sau khi mô hình đã tìm proxy. Bộ nhớ ảnh tự nhiên làm lỗi vòng đầu đặc biệt nghiêm trọng khi query là sketch. Phương pháp bổ sung cho vòng đầu dùng likelihood qua hai bank từ có mô tả miền. [Điểm đầy đủ và KD-SBIR replay](results/freedom-gallery-transfer/summary.json) · [Kiểm tra proxy trên từng split](https://github.com/HieuNM1804/training-free-ZS-SBIR/blob/research/map-retrieval-analysis/docs/results/map-followup/freedom-proxy-audit.json)
