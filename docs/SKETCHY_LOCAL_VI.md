# Truy hồi sketch–ảnh training-free trên RTX 4060 Laptop

## Kết quả local đã kiểm chứng

154 cấu hình và đối chứng đã được chấm đầy đủ trên hai split. Có 137 cấu hình đủ điều kiện để chọn sau khi bỏ các phép xáo ngẫu nhiên và những đối chứng chỉ dùng để chẩn đoán. Cấu hình chung tốt nhất dùng tám view, dành 35% trọng số kênh cho từ vựng và dùng rủi ro chuyển miền mũ 4 để cân riêng phiếu thị giác. [Bảng kết quả trên máy local](results/sketchy-local.csv)

| Phương pháp, cùng cấu hình cho cả hai split | Ext-1 P@100 | Ext-1 mAP@all | Ext-2 P@200 | Ext-2 mAP@200 |
| --- | ---: | ---: | ---: | ---: |
| CLIP ViT-B/32 nguyên bản, chỉ dùng ảnh | 41,64 | 30,29 | 33,84 | 36,34 |
| Chuyển tọa độ, một view, chỉ dùng ảnh | 77,79 | 67,50 | 68,98 | 72,87 |
| Khái niệm công khai gắn với gallery, chỉ dùng khung gốc | 75,48 | 76,17 | 65,08 | 67,76 |
| Chuyển tọa độ hai view và dùng từ điển | 82,29 | 81,05 | 76,79 | 79,16 |
| Năm view, cân riêng phiếu thị giác và từ vựng | 83,07 | 81,32 | 78,45 | 80,46 |
| Tám view, rủi ro mũ 2 | 83,44 | 81,84 | 78,87 | 80,89 |
| **Tám view, rủi ro mũ 4** | **83,46** | **81,83** | **78,96** | **80,96** |
| SeCo-SBIR, cùng ViT-B/32, có huấn luyện | 87,00 | 80,60 | 77,50 | 80,00 |
| MLFA + PCMI, ViT-L/14, có huấn luyện | 89,60 | 85,50 | 82,00 | 79,90 |

Chọn riêng cho Ext-1, cấu hình dùng riêng sai số nguồn và gộp bằng chứng Jaccard đạt **83,76% P@100, 81,88% mAP@all**. Trên Ext-2, cấu hình chung cũng đứng đầu theo precision. So với SeCo-SBIR, cấu hình chung thấp hơn 3,54 điểm P@100 và cao hơn 1,46 điểm P@200. So với MLFA + PCMI, khoảng cách precision tương ứng là 6,14 và 3,04 điểm. [SeCo-SBIR, bảng 1](https://arxiv.org/html/2608.03120v1#S4.T1) · [MLFA + PCMI, bảng 1](https://media.eventhosts.cc/Conferences/ECCV2026/pdfs/9322.pdf)

Giấu mỗi lớp khỏi bước chọn cấu hình trên **cả hai split cùng lúc** cho 83,34% P@100 và 78,45% P@200. Chọn một cấu hình trên các lớp chỉ xuất hiện ở split kia, rồi giữ cấu hình cố định cho toàn bộ split đích, cho 83,39% và 78,57%. Đây là kiểm tra hồi cứu trên không gian phương pháp đã được thiết kế khi thăm dò unseen.

So với hai view, cấu hình chung tăng 1,166 và 2,171 điểm precision. Bootstrap ghép cặp 5.000 lần theo sketch cho khoảng 95% lần lượt từ +0,833 đến +1,508 và từ +1,717 đến +2,632 điểm. Khi lấy lại mẫu theo lớp, hai khoảng rộng hơn, từ +0,340 đến +1,996 và từ +0,123 đến +3,985 điểm. Các khoảng giữ cố định cấu hình đã chọn.

Ở hệ số rủi ro 2, việc ghép đúng độ tin cậy với view tăng khoảng 0,195 và 0,199 điểm precision so với hoán đổi độ tin cậy giữa các tỷ lệ. Chỉ dùng thành phần sai số nguồn cho kết quả nhỉnh hơn dùng cả hai thành phần khoảng 0,011 và 0,021 điểm. Hai khác biệt nhỏ này đều có khoảng bootstrap theo sketch chứa 0. Cặp contour–RGB giúp chuyển miền, còn phần sai số đích bổ sung dùng để cân view chưa cho lợi ích riêng nhất quán. Việc hiệu chỉnh ngoại hình từng ảnh gallery và thêm điểm truy hồi theo chiều ngược cũng không vượt được đối chứng chung của các nhánh đó.

Đối chứng 24 cấu hình trên nhánh tương quan sai số xét thêm hướng sai số của tám view khi chọn trọng số. Dùng cả hai miền với hệ số 0,5 giảm số view hiệu dụng từ khoảng 7,15 xuống 4,28. Biến thể dùng tương quan tốt nhất theo precision trung bình tăng 0,065 điểm P@100 và giảm 0,080 điểm P@200 so với đối chứng giữ nguyên trọng số ban đầu. Khoảng tin cậy bootstrap 95% theo sketch lần lượt là [−0,101, +0,229] và [−0,298, +0,144] điểm. Việc giảm trọng số các view cùng hướng sai số chưa cải thiện precision trên hai split. [Mã nhánh tương quan](https://github.com/HieuNM1804/training-free-ZS-SBIR/tree/8e440f879d235de2cd5ef46436a0d26c7d9db48e)

Nhánh đúng với cấu hình chung và cấu hình riêng tốt nhất dùng commit `3290fe0`. Lượt local tám view chỉ dùng ảnh đã lưu chín cấu hình Ext-1 trước khi chuyển máy theo yêu cầu. Các lượt Colab dùng thư mục kết quả riêng. [Mã phương pháp](https://github.com/HieuNM1804/training-free-ZS-SBIR/tree/3290fe0)

## Thiết lập

Tất cả các phương pháp dùng cùng checkpoint OpenAI CLIP ViT-B/32 đóng băng, SHA-256 `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`. Mỗi sketch chỉ sử dụng các view của chính nó và gallery ảnh không nhãn. Đặc trưng ảnh, các phương trình tuyến tính và đồ thị được tính trong chế độ inference trên GPU local. Baseline là cosine giữa đặc trưng CLIP của sketch và ảnh nguyên khung. Repo KD-SBIR cung cấp thứ tự dữ liệu và hàm chấm.

| Split | Lớp unseen | Sketch | Ảnh gallery | Metric |
| --- | ---: | ---: | ---: | --- |
| Sketchy-Ext-1 | 25 | 14.179 | 17.100 | P@100 và mAP@all |
| Sketchy-Ext-2 | 21 | 12.694 | 12.552 | P@200 và mAP@200 |

Tôi chạy trực tiếp hàm `_retrieval_metrics` lấy từ `D:\distill\src\model.py` trên baseline CLIP RGB. Thứ tự lớp và tệp, cutoff, phép đổi khoảng cosine và phép lấy trung bình đều được đối chiếu. Toàn bộ **53.746 giá trị AP và precision** của 26.873 sketch khớp từng bit giữa hàm gốc và bộ chấm local, kể cả kết quả sau khi lấy trung bình của từng split.

Các cấu hình được thử trực tiếp trên unseen. Khi chọn một cấu hình chung, mỗi split góp một nửa vào trung bình precision. Khi precision bằng nhau, dùng trung bình mAP để chọn. Ở bảng chọn riêng, mAP luôn đi cùng cấu hình đã được chọn theo precision. Việc xác định split theo benchmark dùng danh sách lớp của KD-SBIR. Sau bước này, xếp hạng dùng đặc trưng và chỉ số ảnh. Nhãn được dùng để chấm relevance và chọn tham số thăm dò.

## Tốc độ local và Colab

Trên cùng phép đo gồm 2.048 sketch Ext-1 và 17.100 ảnh, với batch 512, hai nguồn ứng viên và 256 bước đồ thị, Colab L4 mất 0,116 giây so với 0,147 giây trên RTX 4060 Laptop. Ở phần chấm bằng TorchMetrics với bốn worker, local mất 0,973 giây so với 1,527 giây trên Colab. Tăng lên tám worker trên local đưa thời gian này xuống 0,704 giây. Đây là thời gian trung vị của từng công đoạn với đặc trưng đã có sẵn. Cấu hình hai view được chấm lại trên hai máy cho chênh lệch precision tối đa 0,048 điểm phần trăm. [Đo trên RTX 4060](results/sketchy-rtx4060-throughput.json) · [Đo trên L4](results/sketchy-l4-throughput.json)

Danh sách Open Images được cố định trước khi chấm. Đếm sau truy hồi, sau khi đổi dấu gạch dưới trong tên lớp thành dấu cách và chuyển về chữ thường, cho thấy 24/25 tên lớp Ext-1 và 18/21 tên lớp Ext-2 xuất hiện nguyên văn trong từ điển. Lượt chỉ dùng ảnh cho biết hiệu năng khi bỏ hoàn toàn nguồn từ vựng này.

## Cơ chế chuyển miền

Một ảnh gallery tạo ra cặp anchor gồm biểu diễn contour ở miền nguồn và RGB ở miền đích. Để tạo contour, lấy trung bình đặc trưng CLIP của hai ảnh biên Canny với hai mức làm trơn. Hai phía đi qua phép chiếu phổ IsoCLIP. Với từng anchor, dùng các cặp lân cận khác để tái tạo RGB của nó. Chênh lệch giữa đích thật và dự đoán là vector sai số ngoài mẫu. Các sai số được chia cho độ lớn RMS của cả ngân hàng. [IsoCLIP, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Magistri_IsoCLIP_Decomposing_CLIP_Projectors_for_Efficient_Intra-modal_Alignment_CVPR_2026_paper.html)

Khi có view mới, tìm các anchor nguồn gần nhất. Hệ số của chúng có tổng bằng 1 và tối thiểu hóa tổng của sai số tái tạo ở miền nguồn, hạng ổn định đường chéo và bình phương độ lớn của tổ hợp sai số ngoài mẫu ở miền đích. Áp các hệ số đó lên các anchor RGB để dự đoán biểu diễn ảnh phù hợp với view. Phiên bản hai miền dùng chung trục CLIP còn giữ một phần dư của truy vấn. Phép chuyển hệ số sang miền đích có tiền lệ trong neighbor embedding. Giả thuyết được kiểm tra ở đây là sai số của các cặp tham chiếu chứa thông tin để lựa chọn láng giềng và cân các view. [Super-Resolution Through Neighbor Embedding, CVPR 2004](https://cse.hkust.edu.hk/~dyyeung/paper/pdf/yeung.cvpr2004.pdf)

Ở nhánh có từ điển, CLIP chấm các prompt cố định của 20.931 khái niệm Open Images. Mỗi view sketch dùng năm khái niệm gần nhất, mỗi ảnh dùng 16. Vector từ vựng là trung bình embedding văn bản miền ảnh với trọng số softmax. Mỗi view có một danh sách thị giác và một danh sách từ vựng. Độ chồng Jaccard giữa láng giềng của truy vấn và láng giềng của ảnh cung cấp bằng chứng cho từng ứng viên. Query Specific Fusion cung cấp cơ chế gộp đồ thị và xếp hạng lại. Số **ba paper cơ chế** gồm IsoCLIP, neighbor embedding và Query Specific Fusion. CLIP là backbone chung và Open Images là nguồn từ vựng được báo cáo riêng. [Query Specific Fusion for Image Retrieval, ECCV 2012](https://link.springer.com/chapter/10.1007/978-3-642-33709-3_47)

Để ước lượng độ tin cậy của một view, đặt `R_nguồn` bằng sai số tái tạo bình phương của view chia cho khoảng cách bình phương trung bình tới các láng giềng. `R_đích` là bình phương độ lớn của tổ hợp sai số ngoài mẫu của những láng giềng đó. Trọng số của view trong một kênh tỉ lệ với trọng số ban đầu nhân bằng chứng Jaccard mũ p, rồi chia cho rủi ro mũ β. Cấu hình tách kênh cho phép dùng rủi ro chuyển miền ở kênh thị giác và độ chồng láng giềng riêng của kênh từ vựng.

## Phạm vi áp dụng và cách kiểm tra

Phép chuyển tuyến tính dùng được với hai ngân hàng anchor có cùng danh tính. Phép đọc đích thuần túy cho phép hai phía có số chiều khác nhau. Giữ phần dư của truy vấn yêu cầu hai phía dùng chung hệ trục. Bản Sketchy khai thác contour và giả định phản chiếu ngang giữ nguyên nhãn truy hồi theo lớp. Một cách xây anchor khác, dùng prompt mô tả hai miền và ảnh RGB nguyên khung, được chấm riêng với đúng từ điển đã duyệt. Kết quả thực nghiệm hiện chỉ bao phủ hai split của Sketchy.

Các baseline đóng băng được công bố trong SketchFusion đạt 26,1% P@200 và 25,0% mAP@200 với CLIP ViT-L/14, hoặc 59,2% P@200 và 58,8% mAP@200 khi ghép Stable Diffusion với CLIP. Đây là số trích từ bài báo để tham khảo về trường hợp training-free. Đối chứng thực nghiệm của dự án dùng chính checkpoint ViT-B/32 trong bảng trên. [SketchFusion, CVPR 2025, bảng 1](https://openaccess.thecvf.com/content/CVPR2025/papers/Koley_SketchFusion_Learning_Universal_Sketch_Features_through_Fusing_Foundation_Models_CVPR_2025_paper.pdf)

FreeDom là tiền lệ sử dụng từ vựng công khai cho truy hồi chuyển miền. M-LINE là tiền lệ đưa thông tin miền đích trở lại bài toán xác định hệ số tái tạo. Điểm cần kiểm chứng của phương pháp này là việc sai số dự đoán đích khi giấu từng cặp có giúp xếp hạng sketch–ảnh. Những đối chứng trực tiếp gồm bỏ hiệu chỉnh, hoán đổi độ tin cậy giữa các view cùng sketch, và chỉ dùng riêng sai số nguồn hoặc sai số đích. [FreeDom, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/papers/Efthymiadis_Composed_Image_Retrieval_for_Training-Free_Domain_Conversion_WACV_2025_paper.pdf) · [Iris Super-Resolution Using Iterative Neighbor Embedding, CVPR Workshops 2017](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w6/papers/Alonso-Fernandez_Iris_Super-Resolution_Using_CVPR_2017_paper.pdf)

Đối chứng cùng backbone là SeCo-SBIR có học prompt, với 87,00% P@100 và 77,50% P@200. Một mốc khác theo cùng tên split và cutoff là MLFA + PCMI, dùng CLIP ViT-L/14 có huấn luyện và đạt 89,60% P@100 cùng 82,00% P@200. [SeCo-SBIR, bảng 1](https://arxiv.org/html/2608.03120v1#S4.T1) · [MLFA + PCMI, ECCV 2026, bảng 1](https://media.eventhosts.cc/Conferences/ECCV2026/pdfs/9322.pdf)
