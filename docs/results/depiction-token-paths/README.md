# Can thiệp vào biểu diễn danh từ trong CLIP trên ba phép đánh giá đầy đủ

Khi chuyển cách mô tả cùng một vật từ “sketch” sang “photo”, CLIP có thể đổi cả ý nghĩa nó gán cho tên vật. Chúng tôi chạy hai câu tới cùng một lớp, đổi các trạng thái token của riêng tên vật giữa hai câu, rồi cho những lớp còn lại chạy tiếp. Thay cả hai chiều cho phép đo phần biến thiên đi qua danh từ và phần qua ngữ cảnh. Từ điển gồm 20.931 tên Open Images. Một cấu hình dùng chung được chọn theo trung bình mAP của hai split Sketchy và khóa trước khi chấm TU-Berlin. [Phương pháp và nguồn](../../DEPICTION_TOKEN_PATHS.md) · [Cấu hình đã khóa](freeze.json)

| mAP / precision (%) | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin 30 lớp |
|---|---:|---:|---:|
| Mốc khớp từ và vùng | 84,37 / 83,85 | 77,01 / 74,84 | **80,07 / 80,92** |
| Thay danh từ hai chiều, cấu hình chính | **84,43 / 83,77** | 77,47 / 75,20 | 79,69 / 80,60 |
| Thay một chiều, cùng hệ số | 84,39 / 83,76 | 77,36 / 75,09 | 79,80 / 80,84 |
| Thay một chiều, chọn hệ số riêng trên Sketchy | 84,17 / 83,34 | **77,72 / 75,58** | 79,53 / 80,67 |
| Thay theo hướng của cả prompt, chọn hệ số riêng | 84,20 / 83,52 | 77,39 / 75,23 | 79,80 / 80,72 |
| Đối chứng chiếu lên hướng chung, chọn riêng | 84,36 / 83,68 | 77,39 / 75,22 | **80,21 / 81,13** |

Can thiệp chính dùng lớp 11, hệ số −6 cho hiệu ứng qua tên và 0 cho phần còn lại, áp dụng cho cả sketch toàn cục và vùng. So với nền, mAP thay đổi +0,07, +0,46 và −0,38 điểm. Khoảng tin cậy 95% bootstrap theo lớp tương ứng là [−0,35, +0,48], [−0,46, +1,43] và [−1,17, +0,33]. Can thiệp sửa 224 kết quả hạng nhất và làm sai 145 kết quả đang đúng trên Ext-2. Trên TU-Berlin các số này là 31 và 40. Dữ liệu cho thấy cách mô tả có thể đổi hạng ảnh, còn lợi ích chuyển bộ của phép can thiệp hai chiều chưa được chứng minh. [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU-Berlin](tuberlin.json)

Mỗi điều kiện cuối đã được chấm bằng hàm gốc KD-SBIR cho 14.179, 12.694 và 2.130 sketch, tương ứng gallery 17.100, 12.552 và 27.262 ảnh. Ext-1 và TU dùng mAP toàn gallery cùng P@100. Ext-2 dùng mAP@200 cùng P@200. So sánh tham số trên Sketchy là thăm dò trên unseen. Tất cả kết quả dùng cùng OpenAI CLIP ViT-B/32 đóng băng, từng sketch độc lập. [Dữ liệu từng truy vấn](per-query.zip) · [Kết quả pilot](PILOT_VI.md)
