# Các mốc công bố cho truy hồi theo lớp

Mỗi ô là mAP / precision (%). Ext-1 và TU-Berlin dùng mAP@all / P@100. Ext-2 dùng mAP@200 / P@200. Dùng các hàng **categorical ZS-SBIR** của bài gốc. Cột Ext-1 và Ext-2 được đặt cạnh nhau theo tên split. Phạm vi này cho phép đối chiếu độ lớn kết quả trong tài liệu.

| Phương pháp | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin |
|---|---:|---:|---:|
| SpLIP, ECCV 2024 | 80,20 / 86,70 | 76,40 / 77,30 | 73,10 / 78,20 |
| Liu và cộng sự, ECCV 2026 | 85,50 / 89,60 | 79,90 / 82,00 | 77,50 / 81,30 |
| Ba thành phần gốc, T4, seed 42 | 84,78 / 84,25 | 77,86 / 75,67 | 81,49 / 82,18 |
| Thay bộ kiểm tra GCWE bằng độ tập trung theo dịch chuyển | 85,09 / 84,60 | 78,41 / 76,17 | 81,87 / 82,98 |

So với số công bố của Liu và cộng sự, biến thể thấp hơn 0,41 và 1,49 điểm mAP trên hai split Sketchy, cao hơn 4,37 điểm trên TU-Berlin. Với precision, chênh lệch lần lượt là −5,00, −5,83 và +1,68 điểm. Cải thiện precision trên Sketchy còn là khoảng cách đáng kể. [Kết quả GCWE và các đối chứng trên cùng T4](https://github.com/HieuNM1804/training-free-ZS-SBIR/tree/f0cb524/docs/results/gcwe-displacement-full)

SpLIP học prompt đa phương thức trên các lớp seen. Liu và cộng sự dùng CLIP ViT-L/14, học prompt và các phép chiếu để gộp đặc trưng từ nhiều tầng. Nghiên cứu hiện tại dùng frozen OpenAI ViT-B/32, từ điển Open Images, thống kê trên gallery không nhãn và xử lý từng sketch độc lập. Các cấu hình đã được phát triển trên hai split Sketchy unseen. TU-Berlin kiểm tra chuyển bộ dữ liệu với cấu hình khóa. Các số của hai bài là số tác giả công bố. Chưa đối chiếu từng ID, từng thứ hạng hay mã evaluator của họ với bản sao KD-SBIR đang dùng ở đây. Bảng này chưa phải bảng xếp hạng SOTA toàn diện. [SpLIP, bảng 1](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03426.pdf) · [Liu và cộng sự, bảng 1, trang 10](https://media.eventhosts.cc/Conferences/ECCV2026/pdfs/9322.pdf) · [Trang hội nghị của bài ECCV 2026](https://eccv.ecva.net/virtual/2026/poster/5193) · [Mã và kết quả T4](https://github.com/HieuNM1804/training-free-ZS-SBIR/tree/4968808/docs/results/regional-additive-full)
