# Sketch chọn mô tả nào, và mô tả ấy tìm được những ảnh nào?

CLIP ViT-B/32 giữ được khá nhiều thông tin về lớp khi so các bản vẽ của **những vật thể khác nhau**. Khó khăn xuất hiện khi dùng ảnh chụp để nhận ra lớp của sketch. Trong phép thử thay riêng phía tham chiếu, độ chính xác chọn lớp tăng từ **56,78% lên 87,33%** ở Ext-1 và từ **44,75% lên 84,05%** ở Ext-2 khi chuyển từ ảnh chụp sang bản vẽ của cùng nhóm vật thể. Các tham chiếu có nhãn chỉ dùng để xác định lỗi. Chưa có phương pháp truy hồi độc lập nào từ nhánh này đạt mục tiêu +4–5 mAP.

## Cách đọc các số

Chạy trên Kaggle với bốn NVIDIA L4, seed 42 và OpenAI CLIP ViT-B/32 đóng băng. Dùng đủ 14.179 sketch trên Ext-1 (25 lớp) và 12.694 trên Ext-2 (21 lớp). Tách toàn bộ 2.500 / 2.100 ảnh nguồn từng được người vẽ khỏi gallery chấm chính. Còn **14.600 / 10.452 ảnh**. Mỗi ô ghi mAP@all / P@100 ở Ext-1 và mAP@200 / P@200 ở Ext-2, theo phần trăm. Mốc Semantic Dot được chấm lại trên đúng gallery đó. Điểm trên gallery đầy đủ ban đầu là **75,35 / 77,10** và **72,66 / 71,07** theo bản đối chứng của dự án. Hai split Sketchy có lớp chung và được dùng để thăm dò. Đợt này chỉ đo Sketchy. [Giao thức chọn mô tả](../../description-gallery-transfer.md) · [Giao thức thị giác](../../description-visual-reference.md)

## Mô tả có chuyển được sang ảnh chưa dùng để chọn?

Cho biết lớp thật của sketch. Mỗi lớp chọn đúng một câu đạt AP cao nhất trên ảnh nguồn có nhãn. Cố định câu rồi xếp hạng các ảnh còn lại. Chính câu ấy cung cấp cả AP và precision. Trong đối chứng lọc từ, loại tên trùng cách viết của lớp và nghĩa WordNet có từ đồng nghĩa trùng tên lớp. Từ gần nghĩa và từ chỉ bộ phận vẫn có thể xuất hiện. Còn 581 / 586 tên trong 601 tên Open Images và 1.080 / 1.088 định nghĩa trong 1.123 câu WordNet khác nhau. [Kết quả Ext-1](sketchy_1-source_membership.json) · [Kết quả Ext-2](sketchy_2-source_membership.json)

| Bằng chứng dùng để tìm ảnh | Sketchy Ext-1 | Sketchy Ext-2 |
|---|---:|---:|
| Semantic Dot, mỗi sketch độc lập | 74,95 / 76,83 | 72,34 / 69,79 |
| Biết lớp, chọn tên công khai trên ảnh tham chiếu | 87,00 / 94,16 | 97,35 / 93,66 |
| Biết lớp, chọn định nghĩa trên ảnh tham chiếu | 84,67 / 96,14 | 87,52 / 82,04 |
| Biết lớp, chọn tên sau khi lọc | 72,45 / 86,15 | 92,23 / 85,60 |
| Biết lớp, chọn định nghĩa sau khi lọc | 77,60 / 91,38 | 84,35 / 77,93 |
| Sketch tự chọn một trong các định nghĩa đã lọc và được chọn cho các lớp | 54,32 / 61,48 | 46,18 / 43,20 |
| Ảnh nguồn tự chọn trong cùng danh sách, đọc cùng prompt với sketch | 68,44 / 78,33 | 68,82 / 64,09 |
| Sketch tự chọn trong danh sách ngẫu nhiên cùng kích thước, trung bình 16 lần | 14,79 / 16,32 | 17,22 / 15,22 |

Danh sách đã chọn có 25 / 21 câu. Khi đọc câu, sketch chỉ nhận các câu trong danh sách. Việc chuẩn bị danh sách đã sử dụng nhãn lớp. Với *windmill*, một lớp có mặt ở cả hai split, cả hai nhóm ảnh nguồn chọn tên *wheel*. Cả hai cũng chọn định nghĩa mô tả một loại cối xay chạy bằng sức người hoặc động vật. Định nghĩa ấy vẫn đứng đầu khi chọn lại trên các ảnh còn lại. Một câu có thể dẫn tới đúng nhóm ảnh dù đặc điểm trong câu khác cơ chế hoạt động của cối xay gió. AP cao đồng thời đo khả năng tách các ảnh của lớp ra khỏi phần còn lại của gallery. [Danh sách và đối chứng Ext-1](menu-sketchy_1.json) · [Ext-2](menu-sketchy_2.json)

Mức tăng mAP của định nghĩa đã lọc khi **biết lớp** là +2,64 / +12,00 điểm tính theo truy vấn. Khoảng bootstrap 95% cho hiệu ứng cân bằng vật thể rồi cân bằng lớp là [−6,43, +12,38] / [−3,58, +24,34]. Khi đảo vai trò và chấm trên ảnh nguồn, mAP của cùng quy tắc là 81,33 / 65,83, so với 77,82 / 72,51 của Semantic Dot trên các gallery tương ứng. Chia ngẫu nhiên khớp chính xác số ảnh mỗi lớp cho mAP 79,33 / 87,41 theo chiều ít → nhiều và 81,77 / 66,54 theo chiều ngược lại. Kích thước gallery, đặc biệt với AP@200, ảnh hưởng đến cách diễn giải số. [Chia ngẫu nhiên Ext-1](sketchy_1-matched_random.json) · [Ext-2](sketchy_2-matched_random.json)

## Sketch phân biệt được hai mô tả nhưng có chọn đúng giữa tất cả các mô tả?

Chọn đối thủ dễ nhầm nhất của từng lớp từ một nửa vật thể rồi kiểm tra trên nửa khác. Cân bằng mỗi vật thể và mỗi lớp. Với các định nghĩa đã lọc, AUC phân biệt hai lớp từ sketch là **96,97% / 86,07%**, từ ảnh nguồn là **99,71% / 98,59%**. Một độ lệch cố định giữa hai câu giữ nguyên AUC. Với ngưỡng gốc, độ chính xác cân bằng trên cặp khó là 85,08% / 70,01%. Học ngưỡng từng cặp từ ảnh nguồn thuộc nửa khác cho 89,39% / 71,06%. Tuy vậy, tỷ lệ chọn được đúng câu thắng *tất cả* các đối thủ trên toàn bộ mẫu giảm từ 66,31% / 48,83% xuống 62,14% / 35,94%. Chiếu các ngưỡng ảnh về một điểm hiệu chỉnh cho mỗi câu cho mAP 52,63 / 34,37. Dữ liệu chưa ủng hộ dùng hiệu chỉnh ấy để truy hồi. [AUC Ext-1](bias-sketchy_1.json) · [Ext-2](bias-sketchy_2.json) · [Ngưỡng Ext-1](threshold-sketchy_1.json) · [Ext-2](threshold-sketchy_2.json)

## Thay riêng miền của những vật thể tham chiếu

Chia mỗi lớp thành hai nửa, mỗi nửa 50 vật thể nguồn. Mọi bản vẽ của cùng vật thể nằm trong một nửa. Trọng tâm bản vẽ được tính bằng cách lấy trung bình trong mỗi vật thể, chuẩn hóa, rồi lấy trung bình đều giữa các vật thể. Nhận dạng từng sketch trên nửa còn lại. Giữ **cùng một trọng tâm ảnh RGB** làm truy vấn để xếp gallery sau khi chọn lớp. Tất cả ảnh dùng để tạo trọng tâm đều nằm ngoài gallery chấm. Độ chính xác chọn lớp cân bằng vật thể và lớp. Hai cột truy hồi lấy trung bình theo sketch như evaluator gốc. [Ma trận đủ 8 × 5, Ext-1](visual-summary-sketchy_1.json) · [Ext-2](visual-summary-sketchy_2.json)

| Cách chọn lớp để lấy trọng tâm RGB tìm ảnh | Đúng lớp Ext-1 / Ext-2 (%) | Truy hồi Ext-1 | Truy hồi Ext-2 |
|---|---:|---:|---:|
| Được cung cấp lớp thật | 100,00 / 100,00 | 90,11 / 99,97 | 96,80 / 91,54 |
| Sketch so với trọng tâm ảnh RGB của mỗi lớp | 56,78 / 44,75 | 53,20 / 56,75 | 44,65 / 42,75 |
| Sketch so với trọng tâm bản vẽ của cùng các vật thể tham chiếu | 87,33 / 84,05 | 79,15 / 86,75 | 80,99 / 76,72 |
| Ảnh nguồn của truy vấn so với trọng tâm ảnh RGB | 97,72 / 96,57 | 88,35 / 97,75 | 94,10 / 89,34 |
| Sketch so với trọng tâm ảnh đã tráo nhãn, giữ số ảnh mỗi nhóm | 9,78 / 2,48 | 13,20 / 9,67 | 2,44 / 2,26 |

Đổi ảnh tham chiếu sang bản vẽ của cùng vật thể làm tăng **25,95 / 36,34 điểm mAP** theo truy vấn, với khoảng 95% cho hiệu ứng cân bằng vật thể và lớp [16,92, 36,20] / [25,10, 49,51]. So riêng hàng dùng bản vẽ với Semantic Dot, chênh lệch theo truy vấn là +4,19 / +8,65 điểm. Khoảng cân bằng vật thể và lớp là [−2,92, 14,30] / [1,36, 16,69]. Muốn biến phát hiện thành phương pháp, phải tạo được bằng chứng nhận dạng kiểu này từ **một sketch và gallery không nhãn**. Những hàng dùng nhãn, ảnh nguồn hay các bản vẽ tham chiếu đều chỉ có giá trị chẩn đoán.

Kiểm tra hậu nghiệm xác nhận hai split dùng lại **đúng 2.231 sketch của 400 vật thể** thuộc bốn lớp *scissors, tree, wheelchair, windmill*. Giữ nguyên mọi ảnh và mọi thứ hạng, chỉ bỏ các truy vấn thuộc bốn lớp chung khỏi phép tổng hợp. Chuyển tham chiếu từ ảnh sang bản vẽ vẫn tăng mAP ở **19/21 lớp riêng của Ext-1 và 15/17 lớp riêng của Ext-2**. Chênh lệch cân bằng vật thể và lớp là +26,00 [15,77, 36,90] / +39,26 [25,59, 52,65] điểm. Hai trong số các ngoại lệ là *cow* (−9,45) và *seagull* (−17,83) trên Ext-2. Kết quả đặt giới hạn cho cách diễn giải về sự mơ hồ của sketch giữa các lớp. Các lớp riêng vẫn đến từ Sketchy. [Kết quả từng lớp](cross-split-classes.json) · [Mã phân tích](../../../tf_sbir/description_reference_overlap.py)

Việc lựa chọn tên lớp cho VLM và sử dụng đại diện lớp đã có tiền lệ. Giá trị của phép đo ở đây là tách bước nhận dạng của sketch khỏi bước truy hồi ảnh, đồng thời giữ độc lập các vật thể giữa hai phía. Độ mơ hồ của bản vẽ cũng đã được nghiên cứu trực tiếp. Nhánh này chưa cung cấp contribution phương pháp hoặc phép kiểm tra chuyển bộ. [Learning to Name Classes, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Parisot_Learning_To_Name_Classes_for_Vision_and_Language_Models_CVPR_2023_paper.pdf) · [Prototypical Networks, NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/hash/cb8da6767461f2812ae4290eac7cbc42-Abstract.html) · [Modeling the Visual Ambiguity of Human Sketches, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhou_Modeling_the_Visual_Ambiguity_of_Human_Sketches_CVPR_2026_paper.pdf)

## Tái lập

SHA-256 của evaluator KD-SBIR là `9310e11f47f6fafb54c64060709df8a302165e54e2a8812c97bdb85193c15139`. Chạy trực tiếp hàm KD cho các câu được chọn và phát lại bảng hạng chênh tối đa 0,0181 điểm mAP trong toàn bộ thử nghiệm chia gallery, tối đa 0,0078 trên hai phép chia chính. Precision trùng khớp. Các ô dùng trọng tâm được tính trực tiếp bằng hàm KD. Mỗi gói sau chứa ID, các tensor theo truy vấn hoặc vật thể và JSON có commit, thiết bị, seed cùng hash cho từng tensor.

| Phép thử | Commit mã | Gói kết quả | SHA-256 gói |
|---|---|---|---|
| Chuyển câu giữa hai gallery | `e26544f` | [Kết quả](description-transfer-L4.zip) | `0ecb4e8de9009924c4aeec8d9e1287058bb59e734b54321bb3dd24e21a386b22` |
| Sketch tự chọn câu | `f120aeb` | [Kết quả](menu-L4.zip) | `6ebcd1be296ec9ab8eed11e32cb8d2dbc4fb784959f6d6d0ef3340c8c0a133fc` |
| AUC trên cặp dễ nhầm | `a20e6ec` | [Kết quả](bias-L4.zip) | `3a6ab6013bc041c13bb06c17bb3e0ae52aeb8e064dfcd78fe9e9a3c198cdd03c` |
| Ngưỡng từng cặp | `9de553d` | [Kết quả](threshold-L4.zip) | `919829c5434ec118bc9546129c25db8035874ebe23232eebe1bf02414148d3ae` |
| Tách nhận dạng và truy hồi | `c4d7307` | [Kết quả](visual-reference-L4.zip) | `254505301c136fbf9dbabb2e83c09c71fc5cf489876909cc35e3b569b8d66c20` |
