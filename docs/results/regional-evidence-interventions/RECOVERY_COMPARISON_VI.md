# Khôi phục năm thí nghiệm trên Colab L4

Phiên Colab mới tải lại Sketchy và TU-Berlin, tái tạo embedding từ checkpoint OpenAI CLIP ViT-B/32 và chạy đủ năm nhánh bị ngắt. Mỗi nhánh có ba tệp kết quả, cấu hình đã khóa và archive chứa AP, precision theo truy vấn. Hàm tính metric từ KD-SBIR được đối chiếu trên từng truy vấn. Mọi nhánh dùng cùng từ điển 20.931 mục Open Images, xử lý từng sketch độc lập và chỉ dùng ảnh gallery không nhãn để lập chỉ mục.

| Phương pháp | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin |
|---|---:|---:|---:|
| Khớp từ + diffusion RGB, mốc chung | 81,86 / 81,33 | 74,19 / 72,68 | 79,16 / 80,24 |
| Chọn từ sketch thể hiện trong hai lượt | 82,95 / 82,12 | 75,74 / 73,66 | 78,55 / 79,02 |
| So sánh cặp từ giữa nhóm gallery cạnh tranh | 83,26 / 82,28 | 74,35 / 71,87 | 77,32 / 79,90 |
| Ghép bằng chứng bố cục theo từ | 83,17 / 81,99 | 75,15 / 73,56 | 77,88 / 77,66 |
| Đồng thuận bố cục giữa các tư thế | 84,11 / 83,05 | 75,69 / 74,27 | 78,76 / 78,55 |
| Điều kiện hóa theo phong cách ảnh ứng viên | 83,00 / 82,23 | 75,01 / 73,56 | 77,91 / 77,66 |
| Đối chứng tương phản trực tiếp với patch xáo trộn, hai tư thế | 84,26 / 82,96 | 76,00 / 74,40 | 80,10 / 79,63 |
| Giữ và bỏ cùng vùng sketch, rồi đọc vùng phóng to | 83,99 / 83,38 | 76,63 / 74,45 | 79,18 / 79,58 |
| Đối chứng chỉ phóng to các vùng sketch | 84,00 / 83,45 | 76,43 / 74,17 | 79,50 / 80,13 |

Số liệu là mAP / precision theo phần trăm. Ext-1 dùng 14.179 sketch, 17.100 photo, mAP@all / P@100. Ext-2 dùng 12.694 sketch, 12.552 photo, mAP@200 / P@200. TU dùng 2.130 sketch, 27.262 photo của đủ 30 lớp, mAP@all / P@100. Các cấu hình được thăm dò trên hai split Sketchy unseen và khóa trước TU. Hai dataset đều đo truy hồi sketch → photo.

Cả năm phương pháp chính đều thấp hơn mốc khớp từ + diffusion trên mAP TU. Đối chứng tương phản trực tiếp cao hơn mốc này 0,94 điểm mAP TU. Ở nhánh giữ và bỏ vùng, so với chỉ phóng to, mức thay đổi là −0,007, +0,205 và −0,323 điểm mAP. Ba khoảng tin cậy 95% theo lớp đều chứa 0. [Chọn từ theo sketch](https://github.com/HieuNM1804/training-free-ZS-SBIR/tree/810fcc9/docs/results/selective-concept-listener) · [Cặp từ cạnh tranh](https://github.com/HieuNM1804/training-free-ZS-SBIR/tree/b2e539a/docs/results/ordinal-rival-witness) · [Bố cục theo từ](https://github.com/HieuNM1804/training-free-ZS-SBIR/tree/0dde258/docs/results/layout-conditioned-word-evidence) · [Tư thế](https://github.com/HieuNM1804/training-free-ZS-SBIR/tree/f177293/docs/results/pose-conditional-layout) · [Phong cách](https://github.com/HieuNM1804/training-free-ZS-SBIR/tree/9ccad56/docs/results/candidate-style-counterfactual) · [Can thiệp theo vùng](README.md)

## Khoảng cách với mốc có huấn luyện

Liu và cộng sự dùng ViT-L/14 và huấn luyện 60 epoch. Bảng 1 của họ báo cáo 85,50 / 89,60 ở Ext-1, 79,90 / 82,00 ở Ext-2 và 77,50 / 81,30 ở TU. Lấy cùng một cấu hình giữ/bỏ vùng của chúng ta để đối chiếu, chênh mAP là −1,51, −3,27 và +1,68 điểm. Chênh precision là −6,22, −7,55 và −1,72 điểm. Trên Sketchy, C-DisAmb cũng báo cáo 81,2 mAP@200 với prompt fine-tuning. Đây là các mốc hiệu quả được công bố ở điều kiện huấn luyện khác. [Liu et al., ECCV 2026, bảng 1 và mục 4.2](https://media.eventhosts.cc/Conferences/ECCV2026/pdfs/9322.pdf) · [Zhou et al., CVPR 2026, bảng 1](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhou_Modeling_the_Visual_Ambiguity_of_Human_Sketches_CVPR_2026_paper.pdf)

## Sự khác biệt giữa các bản vẽ của cùng một vật thể

Tất cả 14.179 và 12.694 sketch được nối lại với đúng tên ảnh gốc trong gallery sau khi truy hồi. Hai split có lần lượt 2.500 và 2.100 vật thể nguồn. Ở cấu hình giữ/bỏ vùng, khác biệt giữa bản vẽ của cùng vật thể chiếm 57,1% và 56,6% tổng phương sai AP. Trong 2.398 và 3.224 trường hợp hạng 1 sai và có bản vẽ khác để đối chiếu, 91,5% và 90,8% có ít nhất một bản khác truy hồi đúng lớp. Khi chọn ngẫu nhiên một bản vẽ khác của cùng vật thể, tỷ lệ tương ứng là 60,4% và 51,5%. Các kết quả này mô tả biến thiên quan sát được. Chúng chưa xác định nguyên nhân là phong cách người vẽ, mức giản lược hay độ khó của vật thể. [Thiết kế gốc của Sketchy](https://doi.org/10.1145/2897824.2925954) · [Đối chiếu theo vật thể](within-object-diagnosis.json)

Trên 127 hướng trong không gian điểm từ của CLIP, tỷ lệ phương sai nội bộ do thay đổi vùng nhìn của photo có tương quan hạng 0,669 và 0,485 với tỷ lệ phương sai giữa các bản sketch thật của cùng vật thể. So sánh này dùng gallery để tìm cơ sở tọa độ. ID của bản vẽ chỉ dùng khi kiểm chứng sau truy hồi. Tương quan yếu hơn khi dự đoán riêng cho từng sketch. Dự báo bằng các vùng của chính sketch đạt 0,115 và 0,116, so với 0,003 và −0,001 khi dùng một sketch độc lập trong phép đối chứng hậu nghiệm. [Chẩn đoán phương sai biểu diễn](depiction-mode-diagnosis.json)
