# So khớp sketch–ảnh qua xác suất có chung khái niệm

**Phát hiện giữ được trên cả hai bộ dữ liệu là cách so khớp qua khái niệm chung.** Với cùng một công thức, cùng từ điển và trọng số khái niệm đều, phương pháp đạt **78,94% P@100** trên Sketchy Ext-1, **70,28% P@200** trên Ext-2 và **77,22% P@100** trên TU-Berlin. Cấu hình chọn theo Sketchy ước lượng thêm tần suất khái niệm từ gallery không nhãn. Nó đạt **81,27 / 72,86 / 77,08%** theo cùng thứ tự. Tần suất ước lượng từ Sketchy mang nguyên sang TU cho kết quả thấp hơn nhiều. Thí nghiệm tách được phần nào của phương pháp có thể chuyển giao và phần nào phụ thuộc phân bố ảnh đích.

## Từ lỗi CLIP đến điểm truy hồi

Ở các truy vấn mà cosine RGB của CLIP xếp sai hạng nhất, lấy hai ảnh cụ thể cho mỗi sketch là ảnh sai đứng đầu và ảnh đúng có hạng cao nhất của CLIP. Điểm khái niệm mới đảo thứ tự đúng **7.604/8.979** cặp trên Ext-1, **6.770/8.008** trên Ext-2 và **683/881** trên TU. Ví dụ, ảnh chiếc kéo đứng đầu cho một sketch tuabin gió. Phương pháp mới đưa ảnh tuabin lên đầu và từ *wind turbine* đóng góp hơn 99% vào tổng bằng chứng của cặp đó. Một lỗi ngược lại xảy ra khi sketch gà được ghép với ảnh cá đuối qua cụm *white cut chicken*. Các từ này cho biết vì sao mô hình gán điểm cao cho từng cặp. Nhãn lớp chỉ dùng sau khi xếp hạng để đánh giá. [Cặp ảnh, đóng góp từng từ và các trường hợp bị sửa sai](results/shared-concept-likelihood/generalization-audit.json)

Mỗi kết quả là một tỉ số khả năng giữa hai giả thuyết. Theo giả thuyết thứ nhất, sketch và ảnh cùng được giải thích bởi một khái niệm ẩn. Theo giả thuyết thứ hai, hai khái niệm sinh độc lập. Từ điển là toàn bộ 20.931 tên Open Images. Cho ảnh $x$, miền $d$ và từ $w$, đặt $L_d(x,w)=\exp(\tau\,x^\top t_{d,w})$ với embedding ảnh và text đơn vị. $\tau=100$ là logit scale trong checkpoint. Mô hình giả định các khái niệm có chung độ tập trung trên mặt cầu. Với một sketch $q$, thứ hạng của ảnh $g$ tương đương với

$$
S(q,g)=\sum_w L_{sketch}(q,w)\,P(w\mid g),\qquad
P(w\mid g)=\frac{\pi_w L_{photo}(g,w)}{\sum_v\pi_v L_{photo}(g,v)}.
$$

Thừa số chuẩn hóa còn lại của likelihood ratio chỉ phụ thuộc sketch nên có thể bỏ khi xếp hạng. Hai ảnh không cần chọn cùng một từ ở top 1. Bằng chứng được cộng trên tất cả các từ. Thí nghiệm dùng hai cách đặt $\pi$. Prior đều cho điểm chỉ phụ thuộc vào cặp ảnh. Prior gallery ước lượng một histogram tần suất từ toàn bộ ảnh đích bằng maximum likelihood, giữ cố định encoder, mọi embedding từ và nhiệt độ. Không có số từ top-K, danh mục chọn riêng, crop, contour, đồ thị hoặc tổ hợp các điểm truy hồi.

Ba tiền lệ trực tiếp là phép chuyển miền trong không gian từ rời rạc của FreeDom, tiêu chí xác suất chung so với độc lập trong Bayesian Sets và ước lượng prior đích không nhãn trong nghiên cứu label shift. Cách chuyển sang SBIR ở đây là dùng hai likelihood theo miền và kiểm tra riêng tác dụng của bằng chứng ngữ nghĩa, prior và sự trùng tên lớp. Hiệu quả của tiêu chí xác suất trong retrieval và việc dùng biểu diễn khái niệm đã được nghiên cứu trước đây. Chưa có căn cứ để tuyên bố đây là phương pháp đầu tiên thuộc loại này. [FreeDom, WACV 2025](https://arxiv.org/abs/2412.03297) · [Bayesian Sets, NeurIPS 2005](https://proceedings.neurips.cc/paper/2005/hash/182e6c2d3d78eef40e5dac7da77a748f-Abstract.html) · [Alexandari và cộng sự, ICML 2020](https://proceedings.mlr.press/v119/alexandari20a.html)

| Tài liệu | Hiện tượng hoặc đối chứng gốc | Điều được thử trên ViT-B/32 |
|---|---|---|
| FreeDom | Ảnh truy vấn có thể chuyển miền qua các từ rời rạc. Công trình đánh giá việc chọn từ với thông tin truy hồi bổ sung. | Giữ tất cả khái niệm, chấm cặp theo xác suất. Xáo danh tính giữa hai bộ từ và so prompt có tên miền với từ nguyên văn. |
| Bayesian Sets | Tiêu chí xếp hạng xét xác suất biên của việc chia sẻ cùng nguyên nhân ẩn. Tác giả dùng truy vấn tập hợp và so kết quả hoàn thiện tập với Google Sets. | Dùng một sketch, một gallery ảnh và biến ẩn là khái niệm công khai. So tổng likelihood với biểu diễn bằng vector trung bình. |
| Alexandari và cộng sự | Tác giả so ước lượng label shift bằng maximum likelihood và các cách ước lượng cạnh tranh, đồng thời phân tích ảnh hưởng của calibration. | Giữ nguyên nhiệt độ CLIP, tách prior đều, một bước cập nhật, maximum likelihood, hai nửa gallery và prior chuyển từ nguồn. Không có dữ liệu calibration có nhãn trong phương pháp này. |

Các hướng giải thích từ segmentation, semantic correspondence, phân tích biểu diễn và retrieval đã được ghi theo hiện tượng, đối chứng và phép thử trước đó. [Nghiên cứu về lỗi CLIP gốc](CLIP_RETRIEVAL_DIAGNOSIS.md) · [Từ bảng literature review đến các giả thuyết cạnh tranh](https://github.com/HieuNM1804/training-free-ZS-SBIR/blob/4cc13f7/docs/RESIDUAL_STUDY_LITERATURE_VI.md)

## Kết quả chính thức

Mỗi ô ghi **precision / mAP** theo %. Ext-1 và TU dùng P@100 / mAP@all. Ext-2 dùng P@200 / mAP@200. Mỗi hàng giữ nguyên công thức qua các tập.

| Phương pháp | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin, 30 lớp |
|---|---:|---:|---:|
| CLIP ViT-B/32 gốc, cosine RGB | 41,64 / 30,29 | 33,85 / 36,34 | 46,26 / 29,07 |
| Hiệu kỳ vọng có điều kiện, từ nguyên văn, không cần biết miền | 78,30 / 75,59 | 64,56 / 65,91 | 75,70 / 68,41 |
| Khái niệm chung, **prior đều**, không ước lượng từ bộ ảnh | 78,94 / 75,31 | 70,28 / 71,42 | **77,22** / 73,35 |
| Khái niệm chung, **prior gallery không nhãn**, chọn trước TU | **81,27 / 80,60** | **72,86 / 74,02** | 77,08 / **76,59** |
| Mốc lịch sử chuyển gallery qua prompt và diffusion | 80,14 / 80,85 | 71,59 / 73,16 | 80,61 / 76,96 |

Hai công thức chính của đợt này đã cố định trước khi chấm. Việc chọn giữa chúng chỉ dùng trung bình precision hai split Sketchy rồi xét mAP nếu hòa. Tại thời điểm khóa lựa chọn, cả hai nhánh đều có 0 dòng kết quả TU trong đợt mới. Các điểm TU lịch sử đã có từ trước. Tăng precision của phương pháp được chọn so với CLIP gốc là **39,63 / 39,01 / 30,81 điểm**, tăng ở 24/25, 20/21 và 29/30 lớp. So với mốc lịch sử mạnh, chênh lệch là **+1,13 / +1,27 / −3,54 điểm**. [Khóa chọn phương pháp](results/shared-concept-likelihood/joint-freeze.json) · [Mọi đối chứng và cả hai metric](results/shared-concept-likelihood/fixed-configurations.csv)

## Cơ chế nào được đối chứng ủng hộ?

- **Danh tính khái niệm.** Xáo quan hệ giữa từ bên sketch và từ bên photo, giữ các phép tính khác, làm precision giảm xuống **4,39 / 3,62 / 6,77%**. Điểm chung qua các từ có tác dụng vượt xa độ tập trung của từng phân bố riêng lẻ.
- **Tổng bằng chứng thay cho một vector trung bình.** Dùng vector text trung bình của posterior ảnh để chấm sketch cho **77,44 / 67,84 / 76,86%**. Tổng likelihood giữ nhiều cách giải thích ngữ nghĩa của một ảnh. Mức tăng precision từ phép thay này lớn nhất trên Ext-2.
- **Prior của gallery.** So với prior đều, chênh lệch precision là **+2,33 / +2,58 / −0,15 điểm**. Khoảng bootstrap 95% theo lớp trên TU là −1,83 đến +1,46 điểm. mAP trên TU tăng **3,24 điểm**. Do đó, lợi ích precision của bước ước lượng tần suất chỉ được quan sát trên Sketchy, còn phần so khái niệm giữ lợi ích trên cả ba split. [Ablation ghép cặp theo truy vấn và khoảng bootstrap](results/shared-concept-likelihood/generalization-audit.json)

## Kiểm tra khả năng chuyển giao

1. **Mang prior nguồn nguyên vẹn sang TU.** Ước lượng một prior từ 27.447 ảnh riêng biệt thuộc hợp hai gallery Sketchy, lưu tensor và checksum trước khi có lượt TU mới. Dùng nguyên tensor đó cho TU chỉ đạt **46,68 P / 40,22 mAP**. Phép chuyển trực tiếp này thất bại. Hai cách ước lượng prior từ hai nửa gallery TU không nhãn, sau đó chấm **toàn gallery**, đạt **77,08 / 76,62** và **76,95 / 76,47**. Các kết quả phân nửa là đối chứng đã định trước, không tham gia chọn cấu hình. [Prior nguồn và thời điểm khóa](results/shared-concept-likelihood/strict/source-lock.json) · [Cả ba lượt trên TU](results/shared-concept-likelihood/strict/tuberlin-transfer.json)
2. **Những tên lớp khác với hai split Sketchy.** 24/30 lớp TU có tên không trùng sau khi chuẩn hóa hoa thường, gạch ngang, gạch dưới và khoảng trắng. Nhóm này có 1.700 sketch. Vẫn xếp hạng trên đủ 27.262 ảnh và 30 lớp TU. CLIP gốc đạt **47,86 / 31,25**, prior đều đạt **78,84 / 74,88**, prior gallery đạt **78,08 / 78,09**, còn prior mang nguyên từ Sketchy đạt **40,58 / 34,26**. Phép so theo tên không loại trừ quan hệ ngữ nghĩa gần như *mug* và *cup*. Không coi đây là bằng chứng rằng tên lớp vắng mặt trong dữ liệu tiền huấn luyện CLIP. [Danh sách lớp và phép chấm theo nhóm](results/shared-concept-likelihood/generalization-audit.json)
3. **Nguồn từ vựng.** Danh mục công khai có chồng lấp tự nhiên với tên lớp. Cả hai bản dùng cùng 20.931 từ, không lấy tên lớp benchmark để lọc từ. Tên miền *sketch* và *photo* được cung cấp qua các template cố định. Bỏ thông tin miền, chỉ đưa từng từ nguyên văn vào encoder, cho **77,51 / 69,26 / 73,22%** với cùng phép likelihood. [Giao thức và nguồn dữ liệu của mô hình](SHARED_CONCEPT_PROTOCOL_VI.md)
4. **Thay gallery quanh cùng một cặp.** Với prior đều, lấy một sketch và tám ảnh mỗi lớp rồi tính lại điểm cho các cặp khi giữ toàn gallery hoặc chỉ giữ những ảnh được chọn theo thứ tự đảo ngược. Chênh lệch điểm tuyệt đối lớn nhất dưới $6\times10^{-7}$. Biến thể này tăng precision so với cosine gốc ở **24/25, 21/21 và 28/30 lớp**. Phép thử xác nhận điểm cho một cặp không cần ước lượng tần suất hay học lại bất kỳ thành phần nào khi gallery thay đổi. [Kết quả số và khoảng tin cậy](results/shared-concept-likelihood/pairwise-invariance.json)

## Hướng bị loại và khoảng cách tới kết quả công bố

Chuẩn hóa mean và variance của patch trước transformer làm đầu ra hầu như bất biến với phép biến đổi affine nhân tạo theo kênh, nhưng P giảm xuống **4,21 / 5,01 / 4,21%**. Trên Ext-1, trong các sketch sai top 1 của CLIP, ảnh sai chỉ gần sketch hơn ảnh đúng theo khoảng cách moment ở 2,81% trường hợp. Phải giữ lại thông tin mà phép chuẩn hóa làm mất. Bốn nhánh trước đó gồm chọn 64 từ, phối hợp view, phân cụm và vùng từ patch đã được lưu làm kết quả loại bỏ đối với yêu cầu chuyển giao. [Thí nghiệm chuẩn hóa patch](https://github.com/HieuNM1804/training-free-ZS-SBIR/tree/research/patch-style-invariance) · [Kết quả các nhánh bị loại](https://github.com/HieuNM1804/training-free-ZS-SBIR/blob/4cc13f7/docs/SKETCHY_VOCABULARY_RESULTS_VI.md)

Một phép thử bổ sung có điểm cho từng cặp cố định thay tích posterior bằng hệ số Bhattacharyya. Precision rơi xuống 67,35% và 57,88% trên hai split Sketchy. Giả thuyết này dừng trước TU theo điều kiện đã công bố, không tham gia chọn phương pháp chính. [Kết quả và điều kiện đánh giá](https://github.com/HieuNM1804/training-free-ZS-SBIR/tree/research/semantic-posterior-geometry)

Trên cùng backbone ViT-B/32, SeCo-SBIR có huấn luyện công bố P là **87,0 / 77,5 / 84,7%**. Phương pháp prior gallery còn cách **5,73 / 4,64 / 7,62 điểm precision**. mAP lần lượt là 80,60 / 74,02 / 76,59 so với 80,6 / 80,0 / 78,7 trong paper. Đây là đối chiếu giữa các nghiên cứu. Hai bộ dữ liệu chỉ kiểm tra chuyển giao trong bài toán sketch → photo, chưa kiểm tra một cặp miền thị giác khác. [SeCo-SBIR, Bảng 1](https://arxiv.org/html/2608.03120v1#S4.T1)

Mọi thực nghiệm chạy trên Colab L4 với cùng OpenAI CLIP ViT-B/32. KD-SBIR cung cấp giao thức nạp dữ liệu và metric. Hàm gốc khớp từng truy vấn với evaluator khi chấm cùng thứ hạng của cả **29.003 sketch**. [Kiểm chứng đầy đủ](results/shared-concept-likelihood/generalization-audit.json) · [Checksum hàm gốc](results/shared-concept-likelihood/kd-source.json) · [Mã tính likelihood](../tf_sbir/shared_concept_likelihood.py)
