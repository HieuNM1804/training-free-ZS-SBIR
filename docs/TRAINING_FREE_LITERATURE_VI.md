# Training-free CLIP và VLM cho categorical ZS-SBIR

Khảo sát tập trung vào việc chuyển cơ chế từ các bài toán khác sang truy hồi một sketch trong gallery ảnh không nhãn. Điều kiện của các thí nghiệm là encoder đóng băng, không backpropagation, không học prompt, không dùng DINO và chỉ dùng tên lớp của benchmark trong loader và evaluator. Thống kê, láng giềng và đồ thị được xây từ ảnh gallery. Nhánh ngôn ngữ lấy danh mục Open Images V7 đầy đủ từ nguồn công khai, cố định trước khi truy hồi.

## Các paper ZS-SBIR được cung cấp

- **CLIP for All Things ZS-SBIR** dùng prompt dành cho sketch để thu hẹp khoảng cách hai miền. Phần fine-grained bổ sung ràng buộc giữa các lớp và xáo patch để khai thác tương ứng cấu trúc. Hai hiện tượng cần tách khi đánh giá là lệch miền của embedding toàn ảnh và sai khớp ở cấp bộ phận. [CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Sain_CLIP_for_All_Things_Zero-Shot_Sketch-Based_Image_Retrieval_Fine-Grained_or_CVPR_2023_paper.html)
- **SpLIP** cho prompt học được trao đổi hai chiều giữa encoder thị giác và text ngay khi CLIP đóng băng. Phiên bản training-free có thể khai thác trao đổi thị giác–ngôn ngữ qua truy vấn và đặc trưng gallery ở tầng xếp hạng. [ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/3426_ECCV_2024_paper.php)
- **SeCo-SBIR** chuyển ngữ nghĩa từ text encoder vào visual encoder qua các hàm ghép prompt học được và dùng nhất quán qua nhiễu để giữ khả năng tổng quát hóa. Câu hỏi có thể chuyển sang inference là bằng chứng thị giác nào đủ nhất quán để giữ một giả thuyết ngôn ngữ trong quá trình xếp hạng. [SeCo-SBIR](https://arxiv.org/abs/2608.03120)
- **KD-SBIR** chưng cất thứ hạng ảnh và liên hệ ảnh–ngôn ngữ từ teacher vào CLIP ViT-B/32. Bản thảo cung cấp mốc hiệu năng cho mô hình đã huấn luyện. Repo đi kèm cung cấp quy ước loader và evaluator cho các thí nghiệm ở đây.

## Truy hồi xuyên miền và căn chỉnh hình học

| Công trình | Cơ chế và ý nghĩa cho SBIR |
| --- | --- |
| [ICFRR](https://arxiv.org/abs/2303.17703) | Dùng cấu trúc gallery–gallery để truy hồi các ảnh cùng loại mà điểm query trực tiếp bỏ sót. Là tiền lệ trực tiếp trên SBIR. Phải đối chiếu cả backbone đóng băng và phương pháp rerank, vì kết quả hệ thống công bố có thành phần representation được huấn luyện. |
| [Fast Spectral Ranking](https://openaccess.thecvf.com/content_cvpr_2018/html/Iscen_Fast_Spectral_Ranking_CVPR_2018_paper.html) và [Efficient Diffusion on Region Manifolds](https://openaccess.thecvf.com/content_cvpr_2017/html/Iscen_Efficient_Diffusion_on_CVPR_2017_paper.html) | Thông tin truy vấn được lan truyền qua graph ảnh hoặc vùng. Dùng đường đi nhiều bước trên gallery để nối những ảnh có ngoại hình khác nhau nhưng chung khái niệm. |
| [Total Recall](https://www.microsoft.com/en-us/research/publication/total-recall-automatic-query-expansion-with-a-generative-feature-model-for-object-retrieval/) và [k-reciprocal encoding](https://openaccess.thecvf.com/content_cvpr_2017/html/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.html) | Query expansion và kiểm tra láng giềng thuận nghịch đã có tiền lệ. Chúng là đối chứng cho cập nhật từ láng giềng và đồ thị hai nguồn bằng chứng. |
| [GR-CLIP](https://arxiv.org/abs/2507.19054), [CORAL](https://arxiv.org/abs/1612.01939) và [CSLS](https://arxiv.org/abs/1710.04087) | Căn giữa theo modality, khớp covariance, hoặc hiệu chỉnh vùng lân cận có mật độ cao. GR-CLIP làm đối chứng bậc một. CORAL và CSLS gợi ý các thử nghiệm covariance và hubness nếu sai lệch còn tập trung ở một số nhóm ảnh. |
| [Cross the Gap](https://arxiv.org/abs/2502.04263) | Nghiên cứu chỉ ra hạn chế của cosine trong cùng một encoder khi mục tiêu pretraining là image–text. OTI/OVI trong bài tối ưu pseudo-token hoặc pseudo-patch. Ở đây dùng phép chuyển sang từ thật bằng truy vấn từ điển, phù hợp ràng buộc không gradient. |
| [Distribution Aligned Feature Clustering](https://arxiv.org/abs/2301.06685), [Adapt and Align](https://arxiv.org/abs/2305.05144), [CLAIR](https://arxiv.org/abs/2508.12290) | Các công trình gần về căn chỉnh miền, clustering hoặc dùng CLIP làm tín hiệu bổ trợ. Cần xác định riêng phần dữ liệu và bước huấn luyện trước khi dùng số công bố để so sánh với pipeline đóng băng. |

## Chuyển qua ngôn ngữ và truy hồi tổ hợp

| Công trình | Cơ chế có thể chuyển |
| --- | --- |
| [FreeDom](https://arxiv.org/abs/2412.03297) | Tiền lệ gần nhất cho từ vựng Open Images và memory ảnh gallery. Paper chọn cụm từ bằng tần suất phiếu từ láng giềng ảnh rồi ghép miền đích vào text. Đối chứng SBIR trong code dùng cùng từ điển, k=20 gồm query, n=7 và m=7, photo gallery làm memory, với cùng template ngân hàng từ của các ablation. |
| [CIReVL](https://arxiv.org/abs/2310.09291) và [DualCIR](https://openaccess.thecvf.com/content/WACV2025/html/Zhao_DualCIR_Enhancing_Training-Free_Composed_Image_Retrieval_via_Dual-Directional_Descriptions_WACV_2025_paper.html) | Diễn giải truy vấn bằng ngôn ngữ và dùng nhiều chiều mô tả. Áp dụng SBIR bằng cách giữ các giả thuyết về hình dáng hoặc bộ phận, rồi đối chiếu bằng chứng ảnh thay vì dựa hoàn toàn vào một caption. |
| [Local Concept Reranking](https://arxiv.org/abs/2312.08924) | Đưa khái niệm cục bộ vào reranker của composed retrieval. Là đối chứng gần cho ý tưởng chấm mức độ hỗ trợ từng khái niệm bằng vùng ảnh. |
| [WISER](https://arxiv.org/abs/2602.23029) và [STiTch](https://arxiv.org/abs/2605.21261) | Mở rộng giả thuyết truy hồi, fusion thích nghi hoặc đối sánh có cấu trúc. Trong SBIR có thể kết hợp bằng chứng hình học và từ vựng theo độ đồng thuận, hoặc ghép vùng với khái niệm theo phân bố. |

## Thích nghi lúc suy luận và các bài toán downstream

| Công trình | Cơ chế có thể chuyển |
| --- | --- |
| [TDA](https://arxiv.org/abs/2403.18293) và [Test-Time Retrieval-Augmented Adaptation](https://openaccess.thecvf.com/content/ICCV2025/html/Fan_Test-Time_Retrieval-Augmented_Adaptation_for_Vision-Language_Models_ICCV_2025_paper.html) | Bộ nhớ mẫu truy xuất và bằng chứng giả nhãn là nguồn để hiệu chỉnh dự đoán. Trong SBIR, đơn vị memory có thể là khái niệm được hỗ trợ bởi nhiều ảnh gallery, thay cho nhãn lớp benchmark. |
| [LAME](https://arxiv.org/abs/2201.05718), [Mean-Shift TTA](https://arxiv.org/abs/2507.00462) và [ADAPT](https://arxiv.org/abs/2508.15568) | Làm trơn phân bố dự đoán, tìm mode trong không gian đặc trưng, hoặc cập nhật thống kê likelihood bằng suy luận đóng. Chuyển sang một query độc lập bằng cách xây graph và prototype trên gallery. Phương thức thích nghi trên cả batch query là protocol đánh giá khác và chưa dùng ở đây. |
| [Test-time zero-shot generalization without prompt learning](https://arxiv.org/abs/2405.02266) | Định lượng vai trò của nhiều augmentation mà không học prompt. Gợi ý ensemble nhiều bản render của cùng sketch và giảm trọng số bản có tín hiệu lệch. |
| [CALIP](https://ojs.aaai.org/index.php/AAAI/article/view/25152) và [Tip-Adapter](https://arxiv.org/abs/2111.03930) | Attention không tham số và kernel cache. Tip-Adapter gốc dùng support có nhãn few-shot. Dùng nó làm cảm hứng cho memory giả nhãn, với ablation tách hẳn nguồn hỗ trợ. |
| [Active TTA](https://arxiv.org/abs/2611.09084) và [Survey về label-free VLM](https://arxiv.org/abs/2508.05547) | Phân biệt phương pháp đóng băng trọng số, zero-shot, test-time optimization và phương pháp hỏi nhãn. Các bước sử dụng nhãn bên ngoài hoặc gradient không nằm trong protocol này. |

## Các hướng từ bảng segmentation

- **Tham chiếu thị giác và đề xuất vùng.** ReME/RIM dùng tham chiếu vùng để nối dense visual features và ngôn ngữ. Có thể thay vùng tham chiếu có tên bằng vùng ảnh gallery được hỗ trợ bởi nhiều khái niệm, rồi xem bộ phận nào thực sự xuất hiện trong nét query. [ReME](https://arxiv.org/abs/2506.21233) · [RIM](https://arxiv.org/abs/2404.00262)
- **Sàng lọc theo nhất quán.** FreeCP/S2C2Seg dùng tính nhất quán không gian để lọc lớp. VIP kiểm chứng từ khóa phụ bằng thị giác. SBIR có thể chuyển thành sàng lọc khái niệm hoặc giả thuyết vùng trên mỗi shortlist thay vì lọc danh sách tên lớp. [FreeCP](https://arxiv.org/abs/2508.00557) · [S2C2Seg](https://openaccess.thecvf.com/content/CVPR2026/papers/Qing_S2C2Seg_Semantic-Spatial_Consistency_and_Category_Optimization_for_Open-Vocabulary_Segmentation_CVPR_2026_paper.pdf) · [VIP](https://proceedings.mlr.press/v306/zhu26x.html)
- **Hình học và ngữ cảnh của token.** PEARL dùng Procrustes trong attention và lan truyền Laplacian. Talking to DINO chỉ ra vai trò của feature không gian riêng, còn OV-Stitcher chỉ ra vấn đề mất ngữ cảnh giữa các crop. Với ràng buộc hiện tại, đối chứng khu vực dùng CLIP token và giữ vị trí khi crop. [PEARL](https://arxiv.org/abs/2603.21528) · [Talking to DINO](https://arxiv.org/abs/2411.19331) · [OV-Stitcher](https://arxiv.org/abs/2604.08110)
- **Phản hồi từ dự đoán.** Plug-in Feedback và CLIP as RNN gợi ý lặp để kiểm tra lại vùng hoặc xếp hạng khái niệm. Kết quả SBIR nên ablate phản hồi một bước và nhiều bước vì lỗi ở danh sách đầu có thể bị khuếch đại. [Plug-in Feedback](https://arxiv.org/abs/2508.20265) · [CLIP as RNN](https://arxiv.org/abs/2312.07661)

## Giả thuyết được triển khai

**Đồ thị gallery.** Dùng ảnh gallery làm đỉnh, chuyển tiếp kNN từ embedding ảnh và tùy chọn một nguồn hỗ trợ ghép cặp. Làm trơn các feature gallery trước khi nhận query. Lấy phản hồi bằng centroid mềm của các ảnh đứng đầu trong không gian đã làm trơn. Đối chứng riêng gồm hiệu chỉnh ảnh–contour, chỉ query expansion, đồ thị RGB, có và không nguồn hỗ trợ và phản hồi. Canny trong nhóm này chỉ tạo một dạng quan sát để khởi đầu căn chỉnh.

**Bộ nhớ khái niệm từ vựng mở có kiểm chứng bằng gallery.** Với từ k, tính prototype G_k là tổng embedding ảnh, cân theo xác suất ảnh hỗ trợ từ k, sau đó thêm một prior nhỏ từ text embedding. Suy luận bắt đầu bằng sketch–text, truy xuất một nhóm ảnh sơ bộ, rồi đo mức phù hợp của prototype gallery với nhóm ảnh này để kiểm chứng từ nào đủ đáng tin. Cuối cùng xếp hạng bằng ảnh RGB và các vector khái niệm đã mã hóa cho ảnh. Các ablation tách đối chứng tần suất kiểu FreeDom, chuyển thẳng sketch–text–photo, memory từ vựng, prototype có kiểm chứng và nhánh đồ thị.

**Hợp nhất bằng chứng có điều kiện.** Điểm từ bộ nhớ khái niệm và điểm hình học được tính độc lập. Đồ thị có thể dùng cả tương đồng ảnh và sự tương đồng giữa các mã từ vựng của ảnh, với trọng số lớn hơn cho các cạnh láng giềng thuận nghịch. Có so sánh trọng số cố định với trọng số theo độ đồng thuận của top 64 và độ tập trung tương đối của điểm. Tính novelty cần được đánh giá ở chỗ prototype kiểm chứng và cấu trúc đồ thị kết hợp có bổ sung qua split hay không. FreeDom và ICFRR là hai đối chứng gần. Checkpoint OpenAI CLIP ViT-B/32 được khóa cho các đối chứng và ablation.

Với báo cáo benchmark, cấu hình chọn từ unseen được ghi là thăm dò. Các phương pháp chấm trên cùng full gallery và metric của KD-SBIR. [Open Images V7](https://storage.googleapis.com/openimages/web/download_v7.html)


## Công thức để tái lập và làm ablation

Với mỗi ảnh gallery i, chọn tối đa 16 từ theo cosine với template ảnh, sau đó
tính trọng số softmax r(i,k). Prototype của từ k là
`normalize(sum_i r(i,k) p_i + 0.5 t_k)`, trong đó p_i là embedding ảnh và t_k
là embedding text của từ k. Độ tin của từ tỷ lệ với
`sqrt(support_k / (support_k + 0.5))`. Truy vấn ban đầu là sketch–text.
Nhóm 16 ảnh từ lần truy xuất đầu được dùng như bằng chứng thị giác để kiểm tra
lại từ trước khi lấy 12 từ cuối. Các ablation lần lượt tắt kiểm chứng từ theo
gallery, thay vote tần suất cho vote mềm, và thêm descriptor ngôn ngữ cho ảnh.

Đồ thị có cạnh theo cosine ảnh RGB đã căn giữa, có thể thêm cosine giữa
descriptor từ vựng của hai ảnh. Đường đi bốn bước với xác suất restart 0,1
được tính trên các view gallery, sau đó mới chuẩn hóa từng vector. Nhánh
hình học hiệu chỉnh sketch theo cặp ảnh–contour và dùng top 64 để phản hồi.
Nhánh ngôn ngữ giữ điểm độc lập. Các ablation dùng chính hai nhánh đó với
trọng số cố định hoặc trọng số phụ thuộc độ trùng top 64. Từ vựng công khai
gồm 20.931 mục sau chuẩn hóa và loại trùng theo chuỗi.


Hai kiểm chứng bổ sung phân biệt vị trí áp dụng phản hồi. Với nhánh chỉ dùng ảnh, phản hồi có thể thay đổi vector truy vấn nhiều vòng nhưng mỗi vòng vẫn giữ vector gốc. Đối với Personalized PageRank, softmax trên các ứng viên đầu là phân bố hạt giống của một truy vấn. Khối lượng được truyền trên các cạnh gallery trong ba bước với restart 0,2. Việc này thay đổi trực tiếp điểm ứng viên sau khi nhìn thấy query. Đối chứng diffusion tuyến tính chỉ làm trơn vector gallery trước khi có query.

Thử nghiệm nhiều thang lân cận phân biệt tín hiệu về nhóm ảnh gần và khả năng nối các cụm cùng lớp trên đồ thị rộng hơn. Ba thang có 160, 320 và 640 láng giềng, lấy từ cùng danh sách kNN RGB và chuẩn hóa lại trọng số riêng. Nhánh phối hợp lấy tổng có trọng số của điểm ở hai thang sau hai vòng phản hồi có mốc.

Trong nhánh từ điển được báo cáo riêng, hai tập hạt giống ảnh đến từ đường hình học và đường từ vựng sơ bộ. Posterior từ được tính bằng cách cộng các mã từ của thành viên trong từng tập, với trọng số theo điểm truy hồi. Trộn posterior hai đường với trung bình nhân của chúng để giữ bằng chứng được hai đường hỗ trợ. Chia theo căn bậc hai tần suất từ trên gallery để giảm ảnh hưởng từ quá phổ biến, rồi cộng vào logits ban đầu trước khi lấy khái niệm cuối. Phần này kiểm tra liệu sự đồng thuận ở cấp **từ** có tốt hơn hợp nhất điểm ở cuối pipeline hay không.

Các thử nghiệm tách hai protocol. Nhánh chỉ dùng ảnh chỉ nhận sketch và gallery ảnh. Nhánh từ điển thêm đúng danh sách từ công khai Open Images V7, dùng cùng backbone, split và evaluator. Mọi bước chấm một sketch độc lập với các sketch khác.

## Các cơ chế từ hội nghị được kiểm tra

| Công trình | Cơ chế đã được đánh giá | Điều chỉnh khi chuyển sang ZS-SBIR |
| --- | --- | --- |
| [Cross Modal Retrieval with Querybank Normalisation, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Bogolin_Cross_Modal_Retrieval_With_Querybank_Normalisation_CVPR_2022_paper.html) | Inverted Softmax hiệu chỉnh xu hướng một mục gallery được xếp cao cho nhiều truy vấn. Dynamic Inverted Softmax chỉ hiệu chỉnh truy vấn có top-1 nằm trong tập kích hoạt của querybank. | Querybank gồm contour tổng hợp của ảnh trong chính gallery. So sánh phép chuẩn hóa gốc với ước lượng lấy toàn bộ contour và loại các RGB-neighbor gần từng ảnh khỏi mẫu dùng để ước lượng thiên lệch. |
| [Hubness Reduction with Dual Bank Sinkhorn Normalization, ACM MM 2025](https://doi.org/10.1145/3746027.3755468) | Thuật toán A1 cân bằng khối xác suất theo hai chiều bằng Sinkhorn, rồi lấy hệ số cột để hiệu chỉnh target hubness. | Chuyển riêng A1 với querybank contour. A2 của bài gốc cần thêm target bank. [Bản mở có thuật toán A1](https://arxiv.org/html/2508.02538v1) |
| [MeanShift Test-time Augmentation, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Zanella_On_the_Test-Time_Zero-Shot_Generalization_of_Vision-Language_Models_Do_We_CVPR_2024_paper.html) | Giải luân phiên điểm inlier và mode của các view, dùng độ giống phân bố nhãn để tính affinity. Bài gốc có đánh giá trên ImageNet-Sketch. | Dùng tám view của cùng một sketch và thay phân bố tên lớp đích bằng phân bố trên từ điển Open Images. So sánh mode với lấy trung bình và với sketch gốc. |
| [ClusterMine, WACV 2026](https://openaccess.thecvf.com/content/WACV2026/html/Adaloglou_ClusterMine_Robust_Label-Free_Visual_Out-Of-Distribution_Detection_via_Concept_Mining_from_WACV_2026_paper.html) | Phân cụm ảnh không nhãn, gán concept từ CLIP và lấy từ được nhiều ảnh nhất trong từng cụm. | Phân cụm embedding gallery bằng spherical k-means để giữ phép suy luận không huấn luyện head. Dùng tập từ được bỏ phiếu để tạo bộ nhớ và truy hồi. |
| [STiTch, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Li_STiTch_Semantic_Transition_and_Transportation_in_Collaboration_for_Training-Free_Zero-Shot_CVPR_2026_paper.html) | Conditional Transport đối sánh tập mô tả với tập view ảnh bằng kỳ vọng cosine theo hai chiều. | Thay caption từ MLLM bằng năm concept từ nhánh từ điển của CLIP ViT-B/32. So sánh descriptor ảnh toàn cục, descriptor đã qua đồ thị, descriptor từ và các crop RGB. |
| [ASIF, NeurIPS 2023](https://papers.neurips.cc/paper_files/paper/2023/hash/3186591903d9db31770ad131adb5ceb4-Abstract-Conference.html) và [Training-Free Cross-Modal Alignment, AACL 2026](https://aclanthology.org/2026.findings-aacl.203/) | Ánh xạ đầu vào từ hai miền vào cùng hệ tọa độ bằng độ giống của nó với các cặp anchor. Giữ top-k tọa độ, lũy thừa rồi chuẩn hóa L2. AACL dùng trực tiếp điểm dạng này để xếp hạng ảnh. | Tạo cặp anchor từ RGB và contour của cùng ảnh gallery. Giữ nhánh này chỉ dùng ảnh và đo riêng biểu diễn ASIF, biểu diễn lấy từ gallery đã làm trơn và phép phối hợp điểm. |
| [Multi-Layer Feature Aggregation, ECCV 2026](https://link.springer.com/chapter/10.1007/978-3-032-37032-7_33) | Kết hợp đặc trưng ViT nhiều độ sâu để bổ sung tín hiệu hình học cho embedding toàn cục. | Tách riêng tác động của token các tầng 7, 9 và 11 trên ViT-B/32. Bản thử dùng gộp patch và hiệu chỉnh gallery đóng băng. |
| [ClearCLIP, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06346.pdf) | Dùng query–query self-attention của block cuối, lấy đầu ra attention trước residual và FFN để tăng khả năng định vị của từng patch. | Dùng phép chiếu và trọng số từ checkpoint ViT-B/32 hiện có để mã hóa 49 patch trên mỗi ảnh. Chuyển điểm ngữ nghĩa theo vùng sang Conditional Transport giữa concept của sketch và patch của ảnh. |

### Mốc công bố trên Sketchy

SeCo-SBIR và bản thảo KD-SBIR có số trên cả hai split categorical. Bảng 1 của SeCo-SBIR ghi 80,6 mAP@all và 87,0 P@100 cho Ext-1, 80,0 mAP@200 và 77,5 P@200 cho Ext-2. Bản thảo KD-SBIR đi kèm ghi 81,1 và 87,2 cho Ext-1, 81,1 và 79,1 cho Ext-2. Hai mốc đều dùng thành phần được huấn luyện. [SeCo-SBIR, bảng 1](https://arxiv.org/html/2608.03120v1#S4.T1)

C-DisAmb báo cáo 81,2 mAP@200 và 78,9 P@200 cho Sketchy. UNION báo cáo 82,7 và 79,9 với CLIP-L/14 và 5.000 cặp huấn luyện. UNION mô tả 12.694 ảnh gallery, còn loader hiện tại có 12.552 ảnh ở Ext-2. Các số này dùng làm mốc công bố, với khác biệt về gallery và huấn luyện được ghi rõ. [C-DisAmb, bảng 1](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhou_Modeling_the_Visual_Ambiguity_of_Human_Sketches_CVPR_2026_paper.pdf) · [UNION, mục IV-A và bảng II](https://arxiv.org/html/2511.22253v1#S4)

### Ghép một phần bằng Sinkhorn

[SuperGlue, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Sarlin_SuperGlue_Learning_Feature_Matching_With_Graph_Neural_Networks_CVPR_2020_paper.html) thêm một hàng và một cột dustbin vào ma trận tương đồng để điểm ảnh thiếu đối tác có thể bỏ ghép. Phương trình 8–9 của bài xác định điểm dustbin và khối lượng đích. Trong ablation SBIR, cosine giữa concept của một sketch và các view của từng ảnh cung cấp ma trận đầu vào. Sinkhorn chạy trên ma trận đó với điểm dustbin cố định. Điểm rerank là tổng cosine vượt điểm dustbin, có trọng số bằng khối lượng ghép thật. Đối chứng Conditional Transport dùng cùng concept, view, shortlist và trọng số hợp nhất. Các cấu hình được so sánh trên unseen và thuộc nhánh từ điển Open Images.

### Căn chỉnh hiệp phương sai từ gallery

CORAL căn chỉnh miền bằng cách làm trắng đặc trưng nguồn và đổi hiệp phương sai sang miền đích. Trong SBIR, hai hiệp phương sai được ước lượng từ contour của gallery và từ RGB của chính gallery. Phép biến đổi sau đó được áp dụng độc lập cho từng sketch. Ablation kiểm tra cosine sau CORAL, CORAL với truyền trên đồ thị gallery và kết hợp với căn chỉnh theo láng giềng. Nhánh này chỉ dùng ảnh. [CORAL, AAAI 2016](https://ojs.aaai.org/index.php/AAAI/article/view/10306)
