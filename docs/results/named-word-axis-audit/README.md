# Vì sao truy hồi qua từ có tác dụng, và lỗi nào còn đáng giải quyết?

Trên cả ba giao thức, Semantic Dot phụ thuộc mạnh vào hệ trục gắn với các từ cụ thể khi áp dụng softmax. Từ đứng đầu đôi khi che lấp bằng chứng giúp phân biệt hai sketch, nhưng giải quyết hoàn hảo các ảnh mang riêng từ đó chỉ cho phép tăng **0,19 / 3,73 / 1,47 điểm mAP**. Hướng muốn đạt thêm 4–5 điểm cần tác động cả đến những ảnh có từ đứng đầu khác sketch. Phép đo vẫn để mở khả năng cải thiện bằng cách phân biệt các ảnh bên trong những nhóm từ khác.

## Đối chứng trên toàn bộ unseen

Mỗi ô là mAP / precision (%). Ext-1 gồm 14.179 sketch và 17.100 ảnh, Ext-2 gồm 12.694 và 12.552, TU-Berlin gồm 2.130 và 27.262 trên 30 lớp. Dùng mAP@all / P@100 cho Ext-1 và TU, mAP@200 / P@200 cho Ext-2. Tất cả phép chấm chạy trên Kaggle L4, với seed 42 và hàm KD-SBIR gốc. Mỗi sketch được truy hồi độc lập bằng OpenAI CLIP ViT-B/32 đóng băng. Từ điển có 601 tên Open Images. Cặp prompt kế thừa từ giai đoạn thăm dò trên Sketchy là `a drawing of a {}.` và `a photo of the {}.`.

| Biểu diễn khi tính tích vô hướng | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin |
|---|---:|---:|---:|
| CLIP thị giác, cosine | 30,22 / 41,52 | 36,27 / 33,87 | 28,85 / 45,92 |
| Hai phía dùng logit đã trừ trung bình | 46,86 / 56,49 | 54,12 / 49,58 | 42,66 / 53,55 |
| Chỉ sketch qua softmax | 67,39 / 71,37 | 67,93 / 65,82 | 67,52 / 72,15 |
| Chỉ ảnh qua softmax | 70,34 / 75,86 | 73,41 / 70,38 | 66,64 / 74,96 |
| **Semantic Dot, hai phía qua softmax** | **75,35 / 77,10** | **72,66 / 71,07** | **73,46 / 73,98** |
| Hai phía dùng prompt photo, hai softmax | 72,15 / 74,00 | 69,05 / 66,94 | 73,43 / 74,06 |
| Hai phía dùng prompt drawing, hai softmax | 75,28 / 75,98 | 72,38 / 71,22 | 74,84 / 74,88 |
| Xoay chung hệ trục từ trước softmax | 26,67 / 29,04 | 31,07 / 29,49 | 22,59 / 23,37 |
| Hệ trục xoay, khôi phục đủ 601 giá trị gốc của từng ảnh | 18,95 / 23,50 | 26,63 / 20,97 | 15,81 / 19,03 |

Hai hàng cuối là trung bình **metrics của ba lần xoay riêng**, cùng ba ma trận cho mọi split. Mỗi phép xoay bảo toàn mọi tích vô hướng giữa các logit đã trừ trung bình ở hai miền. Đối chứng cuối giữ đúng dãy xác suất đã sắp của từng sketch và ảnh, bao gồm entropy và độ tin cậy lớn nhất, nhưng đổi tọa độ nhận các giá trị ấy. Hoán vị từ giống nhau cho toàn bộ sketch và ảnh bảo toàn điểm. Chênh lệch khi xoay cho thấy trục từ thật mang cấu trúc quan trọng cho phép đọc phi tuyến này. Thí nghiệm chưa phân biệt được mọi cơ sở có nghĩa khác với từ điển hiện tại. [Giao thức](../../named-word-axis-audit.md)

Với một lượt đọc CLIP, có thể kiểm tra trực tiếp việc đi qua từ có làm mất thông tin hay không. Log xác suất sau khi trừ trung bình bằng logit đã trừ trung bình. Cả hai ma trận text đã trừ tâm đều có hạng **512/512**. Đảo hệ tuyến tính từ xác suất tính theo độ chính xác gốc cho sai số tương đối lớn nhất **0,165%** trên tất cả các lượt đọc. Kết quả áp dụng cho từng lượt gốc, lật, hoặc RGB riêng lẻ. Việc gộp hai lượt của sketch là một phép biến đổi bổ sung.

## Khi hai sketch cùng chọn một từ nhưng cần hai ảnh khác nhau

Với mỗi sketch, xét các ảnh có cùng từ đứng đầu với sketch và có đủ ảnh đúng lẫn sai lớp. Ở mức so sánh một ảnh đúng với một ảnh sai, chỉ dùng xác suất ở từ đứng đầu đạt **81,92 / 72,87 / 70,84% AUC**, còn toàn bộ vector đạt **83,41 / 75,61 / 72,12%**. Phạm vi gồm 65,88%, 62,36% và 73,85% số sketch. Bất kể sketch cụ thể nào chọn từ đó, chỉ dùng từ đầu sẽ cho cùng một thứ tự ảnh, do điểm chỉ bằng xác suất của ảnh nhân với một hằng số dương của sketch.

Phép thử sau buộc hai sketch thuộc hai lớp khác nhau cùng chọn giữa hai ảnh, mỗi lớp một ảnh. Cả bốn cùng có một từ thắng. Bảng ghi xác suất **cả hai sketch đều ưu tiên đúng ảnh thuộc lớp mình**. Chọn tất cả sketch đủ điều kiện, tối đa 64 ảnh cho mỗi ô từ–lớp bằng băm ID cố định. Lấy trung bình đều theo cặp lớp rồi theo từ bên trong cặp lớp. Các panel bao phủ lần lượt 5.607, 6.534 và 828 sketch thuộc 23, 21 và 25 lớp. Mỗi sketch được chấm riêng trước khi ghép để phân tích.

| Nguồn bằng chứng | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin |
|---|---:|---:|---:|
| Từ đứng đầu | 0,00% | 0,00% | 0,57% |
| Toàn bộ 601 từ | 12,26% | 11,65% | 5,79% |
| Đóng góp riêng của 600 từ còn lại | 43,22% | 36,73% | 36,42% |
| Cosine CLIP thị giác | 20,64% | 20,48% | 26,82% |

Khi hai phép so điểm hòa, mỗi quyết định được nửa điểm. Từ đứng đầu có tỉ lệ ghép đúng và ghép ngược bằng nhau trong cả ba split. Với 600 từ còn lại, chênh ghép đúng trừ ghép ngược đạt 39,80 / 30,60 / 33,64 điểm phần trăm. Đây là một phép tách bằng chứng trên các cặp đủ điều kiện. [Giao thức cặp](../../word-collision-attribution.md)

## Trần cải thiện trên mAP thực

Giữ cố định vị trí của mọi ảnh có từ đứng đầu khác sketch. Dùng nhãn thật để sắp tốt nhất tất cả ảnh còn lại, kể cả ảnh ban đầu nằm ngoài top 200. So sánh với thứ hạng gốc của từng truy vấn. Với Ext-2, evaluator chia AP cho số ảnh đúng thực sự xuất hiện trong top 200. Đã xét tất cả số lượng ảnh đúng khả thi trong prefix để tìm trần AP, và tính riêng trần P@200. Vì thế hai trần của Ext-2 có thể đạt ở hai thứ tự khác nhau.

| Phép đo | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin |
|---|---:|---:|---:|
| mAP cao nhất có thể dưới ràng buộc | 75,53 | 76,40 | 74,93 |
| **Dư địa so với Semantic Dot, điểm mAP** | **+0,19** | **+3,73** | **+1,47** |
| Dư địa precision, điểm phần trăm | +0,32 | +0,85 | +3,49 |

Trần này dùng nhãn để lý giải hạn chế của một họ can thiệp. Nó chỉ áp dụng nếu vị trí của các ảnh có từ đứng đầu khác sketch được giữ cố định. [Cách chứng minh và chấm trần](../../word-collision-ceiling.md)

## Các lỗi gợi ý hướng nghiên cứu xuyên miền

Ở 1.955/4.232 lỗi top-1 của Ext-1 (46,20%), 1.898/4.463 của Ext-2 (42,53%) và 245/604 của TU (40,56%), ảnh sai đứng đầu có cùng từ thắng với sketch, còn ảnh đúng lớp cao hạng nhất có từ thắng khác. Trên Ext-2 có 79 sketch hươu cao cổ cùng dạng lỗi cụ thể. Sketch và ảnh bò trả về cùng được CLIP gọi là *horse*, còn ảnh hươu cao cổ đúng lớp được gọi là *giraffe*. Trên TU có 7 sketch khinh khí cầu mà sketch và ảnh đèn cây trả về cùng được gọi là *light bulb*, còn ảnh khinh khí cầu đúng được gọi là *balloon*. Đây là ví dụ theo đúng nhãn benchmark và phản hồi CLIP. Thứ tự từ đơn chưa cho biết CLIP nhầm do hình dáng, bối cảnh hay thiếu thông tin trong bản vẽ. [Ext-1](attribution/errors-sketchy_1.json) · [Ext-2](attribution/errors-sketchy_2.json) · [TU](attribution/errors-tuberlin.json)

## Prior art

- *Cross the Gap*, ICLR 2025, cho thấy truy hồi image–image của CLIP có thể cải thiện khi đi qua phương thức còn lại. Tác giả dùng inversion để khảo sát hình học nội miền. Bước qua text tự nó đã có tiền lệ. [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c15da559b33c0f08a70573804cd5fb22-Abstract-Conference.html)
- *FreeDom*, WACV 2025, chuyển ảnh truy vấn thành từ rời rạc rồi ghép phần biểu diễn miền đích. Đây là đối chứng trực tiếp cho các phương pháp chọn tên rồi dùng text tìm ảnh. [Trang nhóm tác giả](https://cmp.felk.cvut.cz/~efthynik/freedom/freedom)
- *Visual Classification via Description*, ICLR 2023, chấm các mô tả trực quan sinh bởi mô hình ngôn ngữ để phân lớp. *Half-Truths Break Similarity-Based Retrieval*, NeurIPS 2026, cho thấy thêm một thuộc tính hoặc quan hệ sai đôi khi còn làm điểm CLIP tăng, và dùng các mô tả đối chứng khi huấn luyện lại. Hai kết quả giúp thiết kế kiểm tra cho ý tưởng nhận dạng qua mô tả. [Menon và Vondrick](https://cv.cs.columbia.edu/sachit/classviadescr/) · [Kargi và cộng sự](https://arxiv.org/abs/2602.23906)
- *Network Dissection*, CVPR 2017, dùng phép xoay hệ trục để khảo sát tính diễn giải. *SpLiCE*, NeurIPS 2024, phân rã embedding CLIP theo các khái niệm văn bản. Hai công trình đặt giới hạn cho tuyên bố mới về cơ sở từ vựng. [Network Dissection](https://openaccess.thecvf.com/content_cvpr_2017/html/Bau_Network_Dissection_Quantifying_CVPR_2017_paper.html) · [SpLiCE](https://proceedings.neurips.cc/paper_files/paper/2024/hash/996bef37d8a638f37bdfcac2789e835d-Abstract.html)
- *Where Compositionality Hides*, bản thảo arXiv đăng ngày 23/11/2026, tách lỗi xếp cặp và loại hướng nội dung chung giữa hai caption. Đây là prior art rất gần nếu muốn coi việc bỏ thành phần chung là đóng góp. Bản thảo xét image–caption với mô tả ghép cặp. Thí nghiệm ở đây định lượng được mức cải thiện mAP tối đa khi chỉ giải quyết nhóm có cùng từ trong gallery SBIR đầy đủ. [Bản thảo của tác giả](https://arxiv.org/abs/2611.30538)
- Hai mốc CIR bổ sung là *BASIC* (NeurIPS 2025), chấm riêng thị giác và văn bản rồi fusion, và *STiTch* (CVPR 2026), dùng caption do LLM sinh, phép chuyển trong embedding và vận chuyển hai chiều. [BASIC](https://proceedings.neurips.cc/paper_files/paper/2025/hash/6391a02f02dc529cfa3f242fb4343c6a-Abstract-Conference.html) · [STiTch](https://openaccess.thecvf.com/content/CVPR2026/papers/Li_STiTch_Semantic_Transition_and_Transportation_in_Collaboration_for_Training-Free_Zero-Shot_CVPR_2026_paper.pdf)

## Tái lập

Commit chấm cơ sở từ và softmax là `752775c`. Hai chẩn đoán đối chứng ghép cặp và trần mAP ở `2041333` và `86309b0`. Gói đầu chứa per-query metrics, ID, khoảng bootstrap theo lớp, kết quả từng phép xoay và mã băm. Gói sau chứa panel cặp lớp và các trần cho từng sketch. SHA-256 lần lượt là `a94cd041727e156b4a472a03dbd4ec74bf355a72d6bdcf9050987820e7c8c86f` và `72e2b9acee6068c48b9caacfa5563853b5f4808204e62cfcd7cd25630e6594f8`. [Tải đối chứng text](axis-audit-L4.zip) · [Tải phân tích và trần](attribution/word-collision-L4.zip)

Ba bảng dùng hai benchmark cùng chiều sketch → photo. Cấu hình phép xoay và các template được khóa trước khi chấm trên cả ba split. Phân tích ghép cặp và trần mAP được thiết kế sau khi xem kết quả trước đó, kể cả số TU. Chúng là phân tích hồi cứu và không cung cấp một đánh giá chuyển giao độc lập cho phương pháp mới.
