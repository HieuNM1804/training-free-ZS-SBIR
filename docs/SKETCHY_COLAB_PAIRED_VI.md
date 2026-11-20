# Truy hồi sketch–ảnh trên Colab L4

## Kết quả ưu tiên precision

Toàn bộ **149 cấu hình và đối chứng**, với 143 cấu hình đủ điều kiện chọn, đã được chấm trên cả hai split. Các giá trị trong bảng là phần trăm. Cùng một checkpoint OpenAI CLIP ViT-B/32 được dùng cho tất cả các dòng của dự án. [CSV 149 cấu hình](results/sketchy-colab-l4.csv) · [Đối chứng ghép cặp và khoảng tin cậy](results/sketchy-colab-l4-audit.json)

| Cách dùng | Ext-1 P@100 | Ext-1 mAP@all | Ext-2 P@200 | Ext-2 mAP@200 |
| --- | ---: | ---: | ---: | ---: |
| Chỉ dùng ảnh, một cấu hình chung | 82,99 | 82,58 | 78,03 | 79,40 |
| Có từ điển, cùng cấu hình cố định, cân lại cạnh gallery | 83,52 | 81,70 | 78,96 | 80,81 |
| **Có từ điển, cùng bộ chọn dựa trên gallery** | **83,64** | **81,93** | **78,93** | **80,92** |
| **Có từ điển, tuning riêng từng split** | **84,42** | **82,45** | **78,96** | **80,81** |
| SeCo-SBIR, ViT-B/32, có huấn luyện | 87,00 | 80,60 | 77,50 | 80,00 |
| MLFA + PCMI, ViT-L/14, có huấn luyện | 89,60 | 85,50 | 82,00 | 79,90 |

Cùng bộ chọn trên hai split, khoảng cách precision so với SeCo-SBIR là −3,36 và +1,43 điểm. Tuning riêng đưa khoảng cách về −2,58 và +1,46 điểm. Với mốc MLFA + PCMI dùng backbone lớn hơn, cấu hình chọn riêng còn cách 5,18 điểm P@100 và 3,04 điểm P@200. [SeCo-SBIR, bảng 1 và phần triển khai](https://arxiv.org/html/2608.03120v1#S4.T1) · [MLFA + PCMI, ECCV 2026, bảng 1](https://media.eventhosts.cc/Conferences/ECCV2026/pdfs/9322.pdf)

- **Ext-1.** `refine-k256-mix2-s64` dùng 256 láng giềng, trộn hai cách bỏ phiếu với trọng số 1/2 và đặt trọng số cạnh truy vấn là 64. Đây là dòng đạt 84,42% P@100. Nếu mang nguyên cấu hình sang Ext-2, P@200 là 77,58%.
- **Ext-2.** `balance-index3-graph3-text35` dành một nửa trọng số cạnh gallery cho bằng chứng ngữ nghĩa trong cả hai bước và 35% trọng số kênh query cho từ vựng. Nó đạt 78,9574% P@200. Cấu hình dùng ba loại cạnh bằng nhau đạt 78,9521% P@200 và 80,9554% mAP@200. Khoảng cách precision giữa hai lựa chọn chỉ 0,0053 điểm.
- **Một cấu hình cho hai split.** `route-neighbors100-k128-q75` dùng khả năng khôi phục 100 láng giềng RGB của các ảnh gallery. Mỗi sketch đọc chênh lệch chất lượng của hai bộ xếp hạng quanh 128 anchor cho mỗi view của hai view dùng để chọn. Ngưỡng là phân vị 75 của ước lượng leave-one-out trên gallery. Khi đưa kết quả của cả hai split về cùng trọng số, phương pháp đạt 81,282% precision trung bình. Phương pháp cố định tốt nhất trong lượt tinh chỉnh đạt 81,254%.

Giấu từng lớp khỏi bước chọn cấu hình trên cả hai split cho **83,47% P@100 và 77,22% P@200**. Khi chỉ chọn theo những lớp có riêng ở split còn lại, hai mức là **83,08% và 77,62%**. Đây là kiểm tra hồi cứu trên tập cấu hình đã xây dựng trong quá trình thăm dò dữ liệu unseen. [Kết quả chọn lại theo lớp](results/sketchy-colab-l4-class-validation.json)

## Bằng chứng cho cách chọn bộ xếp hạng

Bộ chọn dùng chung chuyển sang cấu hình gộp chung ở 6,09% sketch Ext-1 và 2,65% sketch Ext-2. So với luôn dùng bộ xếp hạng 192 láng giềng và cân riêng hai kênh, precision thay đổi +0,154 và −0,023 điểm. So với chọn ngẫu nhiên với đúng cùng tỷ lệ chuyển bộ xếp hạng, lợi thế là +0,109 và +0,045 điểm. Khoảng 95% theo lớp lần lượt là [−0,014, +0,258] và [−0,020, +0,123]. Kết quả ủng hộ tín hiệu chọn yếu ở mức sketch trên Ext-1. Chưa có đủ bằng chứng về lợi ích nhất quán giữa các lớp hoặc giữa các miền khác nhau.

Trong nhánh này, hai bộ xếp hạng cha được chấm trên từng sketch bằng evaluator chính thức. Mỗi quyết định chọn nguyên một danh sách, nên có thể lấy đúng hàng AP và precision của danh sách đó. Một cấu hình được chạy thêm qua evaluator trên toàn bộ ma trận điểm đã chọn. Cả 26.873 sketch khớp từng bit khi so với phép chọn hàng. Kiểm tra toàn bộ 57 mặt nạ quyết định trên mỗi split cũng xác nhận kết quả chính xác theo từng hàng. Trần chẩn đoán nếu được dùng nhãn để chọn giữa hai danh sách là 86,06% P@100 và 81,59% P@200.

Trong thử nghiệm chỉ dùng độ chồng contour–RGB để chọn số láng giềng, mức nhiệt 8 của cách cân riêng kênh giảm 0,056 và 0,207 điểm so với gộp đều. Khoảng 95% theo lớp trên Ext-2 là [−0,414, −0,026]. Tín hiệu phân vị có liên quan đến độ thuần gallery, nhưng việc dùng nó theo cách này để tăng trọng số cho một bán kính làm giảm precision trên Ext-2.

## Hai điều kiện từ vựng

Mọi lượt dùng cùng OpenAI CLIP ViT-B/32 đóng băng. Gallery gồm ảnh không nhãn. Mỗi sketch chỉ sử dụng các biến đổi của chính nó. Phương pháp giải các hệ tuyến tính trong khi suy luận, không học prompt và không tính gradient. Các hình phản chiếu giữ nguyên nhãn truy hồi theo lớp trong Sketchy. Việc chọn siêu tham số trực tiếp trên unseen là thiết lập thăm dò được người dùng cho phép.

Nhánh chỉ dùng ảnh loại bỏ từ vựng ở cả điểm truy vấn, danh sách ứng viên và các cạnh ảnh–ảnh trong gallery. Nhánh có từ điển dùng 20.931 cụm từ Open Images cùng prompt cố định. Danh sách được lấy toàn bộ từ nguồn công khai trước khi chấm. Kiểm tra hồi cứu thấy 24/25 tên lớp Ext-1 và 18/21 tên lớp Ext-2 có mặt nguyên văn sau khi chuẩn hóa chữ thường và dấu gạch dưới.

Ext-1 gồm 14.179 sketch và 17.100 ảnh, chấm P@100 và mAP@all. Ext-2 gồm 12.694 sketch và 12.552 ảnh, chấm P@200 và mAP@200. Cấu hình chung tối đa hóa trung bình hai precision với trọng số bằng nhau, rồi dùng trung bình hai mAP để phá hòa. Hai số báo cáo cho mỗi split luôn lấy từ cùng cấu hình. Bộ chấm sử dụng hàm TorchMetrics theo đúng KD-SBIR. Lần kiểm tra bằng hàm gốc trên baseline CLIP RGB đã đối chiếu đủ 53.746 giá trị per-query. [Kiểm chứng evaluator và dữ liệu](SKETCHY_LOCAL_VI.md#thiết-lập)

## Đối chứng chỉ dùng ảnh

| Cấu hình trên cùng L4 | Ext-1 P@100 | Ext-1 mAP@all | Ext-2 P@200 | Ext-2 mAP@200 |
| --- | ---: | ---: | ---: | ---: |
| Chuyển tọa độ, một view, cosine | 77,80 | 67,51 | 68,97 | 72,86 |
| Chuyển tọa độ, tám view, cosine | 80,00 | 68,76 | 71,72 | 75,81 |
| Một view, xếp hạng bằng đồ thị | 79,68 | 80,40 | 74,21 | 75,14 |
| Tám view, đồ thị, giữ phần dư với hệ số 1 | 82,50 | 82,35 | 77,99 | 79,17 |
| **Tám view, đồ thị, giữ phần dư với hệ số 1,5** | **82,99** | **82,58** | **78,03** | **79,40** |

Ở hệ số phần dư 1, dùng tám view tăng 2,82 và 3,78 điểm precision so với một view khi cùng xếp hạng bằng đồ thị. Thêm cách xếp hạng đồ thị vào tám view tăng 2,50 và 6,27 điểm so với cosine. Bootstrap theo lớp, ghép cặp cùng sketch, cho khoảng 95% của hiệu ứng đồ thị là [+1,50, +3,53] và [+3,20, +9,71]. Giấu từng lớp khỏi bước chọn tham số vẫn chọn hệ số phần dư 1,5 và khôi phục cả hai precision trong dòng cuối.

Trong đối chứng cố định 64 láng giềng và hệ số phần dư 1, thêm phạt sai số đích tăng 0,24 và 1,04 điểm precision. Khi hoán đổi sai số giữa các anchor, cấu hình ghép đúng thấp hơn 0,61 điểm ở Ext-1 và cao hơn 0,05 điểm ở Ext-2. Khoảng 95% theo lớp đều chứa 0. Bằng chứng về lợi ích của tám view và xếp hạng bằng đồ thị rõ hơn bằng chứng dành riêng cho hạng phạt sai số đích.

Tại cấu hình tám view với hệ số phần dư 1 và trọng số bằng nhau cho ba loại cạnh gallery, đưa từ vựng vào toàn bộ pipeline tăng 0,98 điểm P@100 và 0,96 điểm P@200 so với đối chứng loại toàn bộ kênh từ vựng. Khoảng 95% theo lớp lần lượt là [+0,33, +1,70] và [−1,03, +3,26]. Phép so sánh này đo tác động tổng của nguồn từ vựng ở cả query lẫn gallery.

## Đóng góp được kiểm tra

Mỗi ảnh gallery tạo một cặp tham chiếu gồm contour và ảnh màu có cùng danh tính. Với một sketch, các hệ số tái tạo theo contour được áp lên ảnh màu tương ứng. Mỗi cặp gallery còn có một vector sai số được tính bằng cách giấu chính nó, dùng các cặp khác dự đoán embedding ảnh màu rồi so với embedding thật. Kết hợp sai số ở miền nguồn và miền đích cho phép kiểm tra mức đáng tin của từng phép ghép. Cùng cơ chế có thể áp dụng cho hai miền khác khi có cách xây các cặp tham chiếu tương ứng. Bằng chứng thực nghiệm ở đây chỉ bao phủ sketch và ảnh.

Ba bài cung cấp cơ chế trực tiếp gồm IsoCLIP cho phép chiếu phổ, Neighbor Embedding cho phép chuyển tọa độ giữa các cặp và Query Specific Fusion cho đồ thị lân cận Jaccard. Chọn hệ số có tổng bằng một, kiểm tra sai số ngoài mẫu ở miền đích và thử dùng độ chồng lân cận giữa hai miền là các phần được đánh giá trong dự án. [IsoCLIP, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Magistri_IsoCLIP_Decomposing_CLIP_Projectors_for_Efficient_Intra-modal_Alignment_CVPR_2026_paper.html) · [Neighbor Embedding, CVPR 2004](https://cse.hkust.edu.hk/~dyyeung/paper/pdf/yeung.cvpr2004.pdf) · [Query Specific Fusion, ECCV 2012](https://link.springer.com/chapter/10.1007/978-3-642-33709-3_47)

- **Chỉ dùng ảnh.** Thử số láng giềng, mức phạt sai số đích và lượng phần dư của sketch được giữ lại khi chuyển sang miền ảnh. Các đối chứng dùng một view, tám view, cosine và xếp hạng bằng đồ thị. [Branch và commit](https://github.com/HieuNM1804/training-free-ZS-SBIR/tree/c268df56c1eda43e0ff288abcb9ecb46e5189072)
- **Vị trí đưa từ vựng vào đồ thị.** Đổi trọng số cạnh từ vựng khi làm trơn embedding gallery, sau đó đổi riêng trọng số khi xếp hạng. Tại mỗi đồ thị, thử các trọng số từ vựng của truy vấn. Ngay cả khi cạnh ngữ nghĩa của gallery có trọng số bằng 0, lượt thử vẫn có từ vựng nếu nó dùng từ để lấy ứng viên hoặc xếp hạng sketch. [Branch và commit](https://github.com/HieuNM1804/training-free-ZS-SBIR/tree/7f411f8f9d56479b45054d105c5ac184438574ae)
- **Số láng giềng phù hợp với mỗi truy vấn.** Với mỗi ảnh tham chiếu, đo mức trùng của hai tập láng giềng trong không gian contour và RGB ở năm giá trị k gồm 64, 112, 192, 320 và 448. Loại chính ảnh đó khỏi cả hai tập. Với từng k, đổi kết quả thành phân vị trong gallery. Mỗi sketch đọc các phân vị quanh 32 anchor gần nhất của hai view rồi chọn một k hoặc gộp điểm của năm k. Gộp đều và hoán đổi hồ sơ phân vị giữa các ảnh là các đối chứng trực tiếp cho giá trị của tín hiệu chọn k. [Branch và commit](https://github.com/HieuNM1804/training-free-ZS-SBIR/tree/c2036a38218886e17bc6a49ffd69405f017fa695)
- **Tinh chỉnh cùng phương pháp.** Thử 192, 256, 320 và 384 láng giềng. Tại mỗi mức, nội suy giữa việc cân kênh hình và kênh từ riêng biệt và việc gộp hai kênh theo Jaccard. Hai đầu mút khôi phục hai cách bỏ phiếu ban đầu. Thử thêm hai trọng số cạnh truy vấn ở trung điểm của phép nội suy. [Commit tinh chỉnh](https://github.com/HieuNM1804/training-free-ZS-SBIR/tree/b01ad6b99926e7ce51b040566c8714806b83fbd5)
- **Ước lượng trực tiếp chất lượng xếp hạng từ các cặp gallery.** Chia gallery ngẫu nhiên thành bốn phần. Với mỗi phần, lấy hai contour của từng ảnh làm truy vấn thử, tính phép chuyển tọa độ bằng các cặp thuộc ba phần còn lại, rồi truy hồi trên gallery đầy đủ. Ảnh RGB tương ứng cho phép chấm hai bộ xếp hạng theo ba mục tiêu tự giám sát gồm khôi phục 100 láng giềng RGB, khôi phục 200 láng giềng RGB và hạng của ảnh ghép cặp. Với sketch thật, chọn bộ xếp hạng có chất lượng dự đoán cao hơn quanh những anchor gần sketch. Các ngân hàng dùng chung biểu diễn và ảnh gallery. Việc chia bốn phần loại các cặp đang làm truy vấn thử khỏi bước xác định tọa độ. Các trọng số CLIP và danh sách từ vựng đều được cố định. [Branch và commit](https://github.com/HieuNM1804/training-free-ZS-SBIR/tree/1cdcde36e1b29240e364dd15af399a08e6a3fb43)

Bảng literature review về segmentation gợi ý hai câu hỏi có thể chuyển sang truy hồi. S2C2Seg/FreeCP xét tính nhất quán trước khi chấp nhận bằng chứng ngữ nghĩa. PEARL và OV-Stitcher khai thác cấu trúc không gian hoặc nhiều biểu diễn của cùng ảnh. Ở Sketchy, đơn vị có thể đối chiếu là một ảnh gallery xuất hiện trong cả biểu diễn contour và RGB. Việc đo độ tin cậy trên các cặp này gợi mở một phép thử xuyên miền mà vẫn giữ từng sketch độc lập. [Ghi chú từ bảng segmentation](TRAINING_FREE_LITERATURE_VI.md#các-hướng-từ-bảng-segmentation)

Chọn mô hình theo độ chính xác quanh một truy vấn đã có tiền lệ trong phân loại. Chọn và gộp các bộ xếp hạng không giám sát cũng đã được nghiên cứu trong truy hồi ảnh. Đóng góp có khả năng phân biệt của nhánh này là cách tạo tín hiệu so sánh hai bộ xếp hạng từ các cặp hai miền không nhãn và phép thử giấu từng phần của gallery. Cần đối chiếu với cách chọn ngẫu nhiên có cùng tỷ lệ dùng từng bộ xếp hạng để định lượng phần lợi ích thực sự phụ thuộc truy vấn. [Woods và cộng sự, Dynamic Classifier Selection by Local Accuracy](https://academicweb.nd.edu/~kwb/Woods_Kegelmeyer_Bowyer_PAMI_1997.pdf) · [Unsupervised Selective Rank Fusion for Image Retrieval Tasks](https://www.sciencedirect.com/science/article/abs/pii/S0925231219313335)

Sau truy hồi, nhãn gallery cho phép kiểm tra liệu độ bảo toàn láng giềng có liên quan tới độ thuần lớp. Ở k = 192, một phần tư số ảnh có độ chồng thấp nhất có trung bình 89,46% và 87,36% láng giềng RGB cùng lớp trên Ext-1 và Ext-2. Với một phần tư có độ chồng cao nhất, các mức là 98,47% và 95,61%. Hệ số tương quan Pearson trên toàn gallery lần lượt là 0,201 và 0,197. Tín hiệu này liên quan tới độ thuần của một ảnh tham chiếu. Lợi ích khi sử dụng nó để chọn k cho một sketch cần được đo bằng precision trên toàn bộ truy vấn.

## Tái lập trên Colab

Ô Colab cuối trong `training-free.ipynb` clone repo từ GitHub, checkout đúng commit của từng nhánh và chạy hai split Sketchy. Script kiểm tra SHA-256 của checkpoint gốc, tải Sketchy từ Kaggle và ghi kết quả cùng metric theo từng sketch vào `/content/sbir/results`. Đợt này dùng NVIDIA L4 có 23.034 MiB VRAM và 12 CPU. Đặc trưng đã mã hóa được dùng lại giữa các nhánh. Xếp hạng và dựng đồ thị chạy trên GPU, evaluator chạy trên CPU. [Script chạy Colab](../scripts/colab_paired.py) · [Mã kiểm tra và tổng hợp](../tf_sbir/colab_pair_report.py)
