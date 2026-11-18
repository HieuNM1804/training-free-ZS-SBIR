# Training-free CLIP và VLM cho categorical ZS-SBIR

Khảo sát tập trung vào việc chuyển cơ chế từ các bài toán khác sang truy hồi một sketch trong gallery ảnh không nhãn. Điều kiện của các thí nghiệm là encoder đóng băng, không backpropagation, không học prompt, không dùng DINO và chỉ dùng tên lớp của benchmark trong loader và evaluator. Thống kê, láng giềng và đồ thị được xây từ ảnh gallery. Nhánh ngôn ngữ lấy danh mục Open Images V7 đầy đủ từ nguồn công khai, cố định trước khi truy hồi.

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

**Hợp nhất bằng chứng có điều kiện.** Điểm từ bộ nhớ khái niệm và điểm hình học được tính độc lập. Đồ thị có thể dùng cả tương đồng ảnh và sự tương đồng giữa các mã từ vựng của ảnh, với trọng số lớn hơn cho các cạnh láng giềng thuận nghịch. Có so sánh trọng số cố định với trọng số theo độ đồng thuận của top 64 và độ tập trung tương đối của điểm. Tính novelty cần được đánh giá ở chỗ prototype kiểm chứng và cấu trúc đồ thị kết hợp có bổ sung qua split hay không. FreeDom và ICFRR là hai đối chứng gần. Tăng backbone được đo tách khỏi thay đổi phương pháp.

Với báo cáo benchmark, cấu hình chọn từ unseen được ghi là thăm dò. Các phương pháp chấm trên cùng full gallery và metric của KD-SBIR. DFN2B và SigLIP là các nền tảng đóng băng có mục đích kiểm chứng độ nhạy với backbone. [DFN](https://arxiv.org/abs/2309.17425) · [SigLIP](https://arxiv.org/abs/2303.15343) · [Open Images V7](https://storage.googleapis.com/openimages/web/download_v7.html)


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
