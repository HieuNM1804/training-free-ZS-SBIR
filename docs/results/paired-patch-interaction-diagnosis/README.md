# Phản hồi CLIP nào còn phân biệt được vật thể khi đổi miền?

Phép thử trên bản vẽ thật bác bỏ đề xuất dùng riêng tương tác giữa hai nhóm patch làm biểu diễn truy hồi. Một quan sát khác về gallery đáng kiểm chứng hơn. Ngay trong ảnh chụp, mức dịch chuyển embedding theo phong cách đã giúp dự báo một phần độ lệch theo nội dung giữa sketch và ảnh. Việc suy ra độ lệch cho **một sketch và gallery không nhãn** vẫn là một bài toán mở.

## Hai vật thể, hai ảnh và hai bản vẽ độc lập

Mỗi lớp lấy 32 ảnh nguồn khác nhau, mỗi ảnh lấy hai bản vẽ tay. Ext-1 gồm 800 ảnh nguồn và 1.600 sketch. Ext-2 gồm 672 ảnh nguồn và 1.344 sketch. Bốn lớp chung dùng đúng cùng danh tính. Hai bộ ảnh tham chiếu rời nhau, và mọi ảnh tham chiếu đều khác ảnh nguồn. Với mỗi phép chia ngẫu nhiên 49 patch thành hai nhóm A/B, đo hiệu bốn lần qua CLIP. Nếu đáp ứng F có dạng f(A)+g(B), hiệu tương tác triệt tiêu chính xác. Phần cộng tính đối chứng lấy từ cùng bốn đáp ứng. Kiểm tra cho thấy việc ghép patch tại đầu vào ViT khớp với ghép pixel ở độ chính xác của phép đo · sai số lớn nhất trên hàm cộng tính thử nhỏ hơn 1,1 × 10⁻⁷. Khóa phép đo và điều kiện chọn trước khi chạy. [Giao thức](../../paired-patch-interaction-diagnosis.md) · [Mã đo](../../../tf_sbir/paired_patch_interaction_probe.py)

| Tỷ lệ ghép đúng (%) | Ext-1 · khác lớp | Ext-2 · khác lớp | Ext-1 · cùng lớp, đúng ảnh nguồn | Ext-2 · cùng lớp, đúng ảnh nguồn |
|---|---:|---:|---:|---:|
| CLIP thị giác gốc | 98,56 | 97,52 | 71,83 | 69,87 |
| Semantic Dot 601 từ | 99,05 | 98,80 | 56,79 | 56,73 |
| Phần cộng tính | 89,42 | 85,33 | 66,55 | 64,57 |
| Phần tương tác | 56,21 | 62,63 | 53,20 | 54,31 |

Phép ghép khác lớp dùng một ảnh *khác vật thể* cùng lớp với mỗi truy vấn. Xác suất đúng ngẫu nhiên là 50%. Hiệu tương tác trừ cộng tính là −33,20 điểm trên Ext-1 (bootstrap 95% theo lớp [−40,55 · −25,59]) và −22,70 trên Ext-2 ([−29,51 · −15,19]). Sau khi loại bốn lớp chung, kết quả tương tác còn 55,41% / 62,15%, đối chứng cộng tính đạt 89,81% / 84,87%. Riêng ảnh chụp so với ảnh chụp, tương tác ghép đúng 87,41% / 89,93%. Tín hiệu hiện diện trong miền ảnh nhưng suy giảm mạnh khi một vế là sketch. Không có lượt full mAP hay lượt TU-Berlin cho đề xuất này. [Kết quả Ext-1](diagnosis-sketchy_1.json) · [Kết quả Ext-2](diagnosis-sketchy_2.json) · [Mã thống kê và đối chứng](../../../tf_sbir/paired_patch_diagnostic_statistics.py)

Trên **toàn bộ** hàng AP từng truy vấn đã lưu của Semantic Dot, hai bản vẽ của cùng ảnh nguồn chênh nhau trung bình 17,15 / 23,42 điểm AP theo split · mỗi ảnh nguồn có cùng trọng số. Với hai nguồn khác nhau trong cùng lớp, mức chênh là 21,76 / 28,53. Ví dụ ở một truy vấn Ext-2 thuộc lớp windmill, bản vẽ chỉ có cột và vài cánh dài · ảnh kéo vượt ảnh tuabin đứng đầu trong lớp windmill. Những trường hợp như vậy mô tả chính cặp quyết định đã chấm. Tần suất của kiểu hình trực quan này chưa được ước lượng. [Đường dẫn ảnh nguồn, ảnh đúng, đối thủ và bản vẽ](real-errors-sketchy_2.json)

## Các ảnh thực sự dẫn đầu trên toàn gallery

Với mỗi sketch, bỏ đúng ảnh nguồn khỏi gallery, tìm ảnh đúng lớp có điểm cao nhất và ảnh sai lớp có điểm cao nhất. Khi đổi người đọc hoặc tham chiếu, đối chiếu trên **hai ảnh đã cố định**. Các tỷ lệ ở phần này là thắng trên cặp tĩnh, không phải mAP sau khi xếp lại.

| Khi lựa chọn đầu của cosine thị giác sai | Sketchy Ext-1 | Sketchy Ext-2 |
|---|---:|---:|
| Số truy vấn | 9.004 / 14.179 | 7.983 / 12.694 |
| Trung bình các sketch khác của cùng nguồn ưu tiên ảnh đúng | 11,44% | 16,60% |
| Ảnh nguồn RGB ưu tiên ảnh đúng | 93,26% | 87,74% |
| Trừ ưu tiên của các sketch ở những lớp khác, cặp cố định đổi sang đúng | 83,87% | 79,51% |

Nhóm sketch tham chiếu ở hàng cuối loại cả lớp truy vấn và lớp đối thủ. Cách tính dùng nhãn và nhiều sketch sau đánh giá. Trên lỗi do chính Semantic Dot chọn, trung bình các bản vẽ khác của cùng nguồn ưu tiên ảnh đúng ở 63,23% / 60,81% cặp. Hai tập lỗi được chọn bởi hai ranker khác nhau. [Bảng chi tiết Ext-1](real-errors-sketchy_1.json) · [Ext-2](real-errors-sketchy_2.json) · [Mã chọn đối thủ](../../../tf_sbir/paired_gallery_error_audit.py)

### Đối chiếu mAP trên đầy đủ gallery

Mỗi ô là mAP / precision (%). Ext-1 dùng mAP@all / P@100 · Ext-2 dùng mAP@200 / P@200 theo evaluator KD-SBIR. Các nhóm tham chiếu chứa ảnh nguồn khác danh tính với truy vấn. Tất cả hàng bên dưới dùng cosine thị giác, ngoại trừ Semantic Dot được ghi riêng. Ảnh nguồn của sketch nằm trong gallery gốc khi chấm mAP, nhất quán với giao thức sử dụng cho baseline.

| Thông tin dùng để chấm | Sketchy Ext-1 | Sketchy Ext-2 |
|---|---:|---:|
| Semantic Dot 601 từ | 75,35 / 77,10 | 72,66 / 71,07 |
| Cosine thị giác CLIP gốc | 30,22 / 41,52 | 36,27 / 33,87 |
| Trừ vector trung bình của gallery ảnh không nhãn | 39,02 / 41,86 | 36,60 / 34,36 |
| Trừ vector trung bình sketch, loại toàn bộ lớp truy vấn (oracle) | 67,50 / 76,59 | 69,48 / 65,87 |
| Cộng hiệu giữa hai trung bình từ các lớp khác (oracle) | 42,58 / 64,84 | 56,85 / 50,11 |
| Cộng dịch chuyển đã đo của đúng lớp truy vấn (oracle) | 86,03 / 98,83 | 96,46 / 92,80 |
| Chỉ dùng tâm ảnh của đúng lớp truy vấn (oracle) | 90,32 / 99,93 | 97,86 / 94,86 |
| Thay sketch bằng ảnh nguồn đã biết (oracle) | 76,70 / 94,27 | 91,04 / 85,43 |

Các oracle định vị tổn thất và đòi hỏi thông tin ngoài giao thức suy luận. Chúng chưa phải là cận trên về mặt toán học. Khi chỉ cho biết nhãn để gom ảnh gallery, rồi để sketch tự chọn tâm lớp gần nhất, độ chính xác top-1 chỉ đạt 57,58% / 50,36%. Biết lớp của chính truy vấn đem lại lợi thế rất lớn. [Đối chứng full ở Ext-1](offset-oracles-sketchy_1.json) · [Ext-2](offset-oracles-sketchy_2.json) · [Mã chia danh tính](../../../tf_sbir/paired_domain_offset_oracles.py)

## Gallery ảnh có chứa tín hiệu dự báo độ lệch riêng theo nội dung không?

Trên 14.600 / 10.452 ảnh gallery mở rộng tách khỏi toàn bộ 2.500 / 2.100 ảnh nguồn, đo trục phong cách bằng hai prompt chung `a drawing of an object.` và `a photo of an object.`. Trong mỗi lớp, đo thành phần đồng biến thiên giữa embedding ảnh và vị trí trên trục này. Chiếu bỏ chính hướng prompt khỏi **cả biến dự báo và giá trị đích** để tránh tự đo lại trục vừa dùng. Đích là dịch chuyển trung bình sketch → ảnh trên những danh tính ảnh nguồn độc lập.

Giữ lần lượt một lớp ra ngoài. Dùng các lớp khác để ước lượng hệ số hồi quy dùng chung, rồi dự báo vector dịch chuyển của lớp giữ lại. Hai split Sketchy đều là dữ liệu phát triển. Nhãn gallery, danh tính ảnh nguồn và dịch chuyển của các lớp khác được dùng **trong chẩn đoán**. Mức giảm sai số ở đây tính trên tổng sai số bình phương · đây không phải số điểm mAP.

| So sánh | Ext-1 | Ext-2 |
|---|---:|---:|
| Giảm sai số so với dự báo chỉ bằng tâm ảnh | 7,45% | 9,37% |
| Phân vị 95 của mức giảm từ 128 hướng ngẫu nhiên | 2,75% | 6,36% |
| Giảm sai số sau khi đã cho mô hình dùng cả tâm lớp và trục trung bình gallery | 3,68% | 6,07% |
| Giảm sai số sau đối chứng mạnh, loại cả bốn lớp chung | 2,73% | 5,27% |
| Sketch nằm ngoài min–max của **ảnh mở rộng cùng lớp** trên trục prompt | 89,33% | 85,55% |

Biến thiên trong gallery có thông tin về **hướng** của độ lệch. Khoảng giá trị thực sự quan sát được ở ảnh hầu như không bao phủ sketch. Cùng một đường cong trong khoảng quan sát có thể có vô số cách nối dài bên ngoài. Dự báo được một lát ảnh giữ lại là điều kiện cần đối với một mô hình đường biến đổi, nhưng vẫn phải chứng minh riêng nó dự báo được sketch thật. [Phép đo và miền giá trị Ext-1](support/gallery-response-sketchy_1.json) · [Ext-2](support/gallery-response-sketchy_2.json) · [Mã dự báo giữ lại từng lớp](../../../tf_sbir/gallery_depiction_feasibility.py)

## Quyết định nghiên cứu

Hiệu bốn lần đọc chỉ là một tương tác bậc hai chuẩn. CutMix đã ghép vùng giữa các ảnh và các phương pháp quy kết đóng góp đã định nghĩa sai phân bậc cao. Dữ liệu cũng bác bỏ giả thuyết vận hành. Nhánh này dừng ở chẩn đoán. [CutMix, ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Yun_CutMix_Regularization_Strategy_to_Train_Strong_Classifiers_With_Localizable_Features_ICCV_2019_paper.html) · [Functional ANOVA, AISTATS 2020](https://proceedings.mlr.press/v108/lengerich20a.html) · [Shapley–Taylor, ICML 2020](https://proceedings.mlr.press/v119/sundararajan20a.html)

**Ứng viên duy nhất đề nghị kiểm chứng** là nhận dạng cùng một nhóm nội dung qua các lát gallery có mức độ giống ảnh chụp khác nhau. Một mô hình ước lượng đường biến đổi về ngoại hình của nhóm phải dự báo được ảnh trong lát giữ lại. Sau đó kiểm tra xem hướng đã ước lượng có khớp dịch chuyển trên các sketch thật hay không. Tên lớp và sketch khác chỉ dùng để kiểm chứng. Khớp một đường thẳng qua các tâm đã biết, phân cụm đơn lẻ và hồi quy hỗn hợp đều có tiền lệ. Phần có thể trở thành contribution là điều kiện nhận dạng được **cùng nội dung khi cả ngoại hình lẫn tỷ lệ nhóm thay đổi** từ một gallery không nhãn, kèm phép đo khi sự tương ứng hoặc ngoại suy không xác định. Dữ liệu trên chưa chứng minh điều kiện này. [Spectral Experts, ICML 2013](https://proceedings.mlr.press/v28/tejasvichaganty13.html) · [Continuously Indexed Domain Adaptation, ICML 2020](https://cida.csail.mit.edu/)

FreeDom đã truy hồi xuyên miền bằng từ vựng rời rạc và mở rộng qua database. BASIC kết hợp điểm ảnh và văn bản. STiTch tinh chỉnh caption và tính vận chuyển giữa hai tập biểu diễn. QB-Norm hiệu chỉnh độ phổ biến của ứng viên bằng ngân hàng truy vấn. Các công trình này khiến chỉ khử thiên lệch toàn cục hoặc gộp điểm khó đứng thành contribution độc lập. [FreeDom, WACV 2025](https://arxiv.org/abs/2412.03297) · [BASIC, NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/hash/6391a02f02dc529cfa3f242fb4343c6a-Abstract-Conference.html) · [STiTch](https://arxiv.org/abs/2605.21261) · [QB-Norm, CVPR 2022](https://arxiv.org/abs/2112.12777)

CLIP ViT-B/32 OpenAI đóng băng, seed 42. Phép đo patch chia trên bốn GPU L4 tại Kaggle. Toàn bộ suy luận và chấm điểm chạy tại Kaggle. Số liệu thăm dò này chưa cung cấp một method độc lập đạt mục tiêu +4–5 mAP. [Dữ liệu đo theo truy vấn và embedding dùng cho chẩn đoán](paired-readouts-L4.zip)
