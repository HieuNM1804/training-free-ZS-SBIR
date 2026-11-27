# Vì sao lấy trung bình 64 ảnh làm tăng mAP theo lớp?

Trên Kaggle L4, cùng một đồ thị 64 ảnh láng giềng theo cosine RGB được dùng để đọc hai loại vector. Mỗi ảnh gallery luôn được xếp hạng. Bảng dùng toàn bộ truy vấn unseen, CLIP ViT-B/32 đóng băng và seed 42. Mỗi ô là mAP / precision (%). Ext-1 và TU dùng mAP@all / P@100. Ext-2 dùng mAP@200 / P@200.

| Vector của ảnh được lấy trung bình | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin, 30 lớp |
|---|---:|---:|---:|
| Semantic gốc | 75,35 / 77,10 | 72,66 / 71,07 | 73,46 / 73,98 |
| Semantic qua 64 ảnh | 80,15 / 79,94 | 75,71 / 73,60 | 78,12 / 76,50 |
| CLIP thị giác gốc | 30,22 / 41,52 | 36,27 / 33,87 | 28,85 / 45,92 |
| CLIP thị giác qua cùng 64 ảnh | 44,71 / 42,72 | 39,39 / 39,62 | 43,58 / 52,12 |

Cải thiện theo lớp xuất hiện ở cả hai cách đọc. Với Semantic trên Sketchy, loại ảnh nguồn của từng sketch khỏi tập được phép cho mượn vector vẫn cho 80,43 / 80,03 và 75,31 / 73,59. Mỗi lần chỉ một nửa gallery được dùng làm nguồn láng giềng. Toàn bộ gallery vẫn là ứng viên.

## Tách độ dịch tâm lớp và biến thiên bên trong lớp

Sau khi có thứ hạng, dùng nhãn thật để viết mỗi vector gallery thành tâm lớp cộng với phần lệch riêng của ảnh. Đổi hai phần giữa biểu diễn gốc và biểu diễn 64 ảnh theo cả hai thứ tự. Hai hàng cuối ghi đóng góp mAP trung bình qua hai thứ tự. Trên mỗi split, tổng hai đóng góp bằng đúng chênh lệch mAP quan sát được.

| Can thiệp dùng nhãn để phân tích | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin |
|---|---:|---:|---:|
| Chỉ thay tâm lớp, mAP / precision | 76,18 / 77,24 | 72,73 / 71,10 | 73,95 / 73,88 |
| Chỉ thay phần lệch riêng, mAP / precision | 79,43 / 79,79 | 75,63 / 73,61 | 77,45 / 76,41 |
| Khớp tâm và tổng độ phân tán, giữ thứ hạng nội lớp, mAP / precision | 80,46 / 78,24 | 74,08 / 72,64 | 78,42 / 75,14 |
| Đóng góp của thay tâm lớp, điểm mAP | +0,78 | +0,07 | +0,58 |
| Đóng góp của thay phần lệch riêng, điểm mAP | +4,02 | +2,97 | +4,08 |

Trong cách phân rã này, thay phần lệch riêng giải thích 84%, 98% và 88% mức tăng mAP. Phép chỉ co độ phân tán tái hiện được mAP trên Ext-1 và TU, nhưng thấp hơn trên Ext-2 và cho precision thấp hơn trên cả ba split. Ba can thiệp ở đầu bảng cần nhãn gallery và chỉ có giá trị chẩn đoán.

## Phân biệt hai ảnh nguồn cùng lớp

Trên Sketchy, chọn hai vật thể khác nhau trong cùng lớp cùng hai bản vẽ có ảnh nguồn tương ứng. Giữa hai cách ghép, đo tỉ lệ CLIP ưu tiên đúng cặp theo tổng điểm. Hai ảnh nguồn đều nằm ngoài tập cho láng giềng. Tính trung bình theo vật thể và lớp, cho nửa điểm khi hòa. Hai con số trên Ext-1 là 58,85% với Semantic gốc và 57,82% sau khi dùng 64 ảnh. Trên Ext-2, kết quả là 58,07% và 56,60%. Khoảng bootstrap 95% theo lớp của chênh lệch là [−1,79; −0,25] và [−2,58; −0,33] điểm phần trăm. R@1 và MRR tìm đúng ảnh nguồn giữa tất cả ảnh cùng lớp lại tăng nhẹ, cho thấy kết luận phụ thuộc vào phép đo danh tính. TU không có ghép cặp bản vẽ–ảnh nguồn để đo hiệu ứng này.

[Giao thức và định nghĩa phép ghép](../../semantic-donor-specificity.md) · [Phép tách theo lớp](../../donor-class-moments.md) · [Dữ liệu Sketchy và ghép cặp gốc](https://patsorn.me/projects/sketchy/)

Mã chạy chỉ mục và chấm từng truy vấn ở commit `31218a7b0858169eebc90d86dedea8ceedb8ba27`. Mã tách tâm lớp ở `d3afeacd5fd48eceddacd749a9010b6032bb390a`. Chín tệp JSON cùng thư mục chứa hai metrics chính thức, chẩn đoán danh tính, khoảng tin cậy và mã băm kết quả. Gói kèm các giá trị từng truy vấn có SHA-256 `cf28168034653ba6311ff001d0de6ade69c289ea976e3a785d7f7df2545a2357`. [Tải gói kết quả](donor-specificity-L4.zip)
