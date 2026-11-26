# Đối sánh ảnh bằng dự đoán đáp ứng khi che hai vùng

## Quan sát dẫn đến câu hỏi

Trong một phép thử trước trên Sketchy, quan hệ giữa các từ CLIP qua các crop nhận ra cùng một ảnh RGB sau lật với độ chính xác 89,89% / 89,42%. Đổi một phía thành sketch của chính ảnh đó, độ chính xác còn 51,56% / 54,50% trong bài toán hai lựa chọn với ảnh nhiễu cùng lớp. Hai tác vụ có cùng cặp ảnh ứng viên. Đây là bằng chứng rằng quan hệ có thể lặp lại trong miền ảnh nhưng yếu khi chuyển qua nét vẽ. Bài toán chẩn đoán ở đây là nhận dạng ảnh nguồn. Chỉ số truy hồi chính bên dưới đánh giá mọi ảnh cùng lớp. [Số liệu trên 900 và 756 sketch](https://github.com/HieuNM1804/training-free-ZS-SBIR/blob/research/paired-concept-response-geometry/docs/results/paired-concept-response-geometry/README.md)

Các thí nghiệm nhận thức về vẽ từ ảnh và vẽ từ tên cho thấy mức chi tiết được giữ lại phụ thuộc vào mục đích biểu đạt. Điều này gợi một câu hỏi kiểm chứng được trên CLIP. Nếu hai miền giữ một cấu trúc nhận dạng chung, sự tương ứng suy ra khi bỏ *một* vùng có dự đoán được cách CLIP đổi nhận định khi bỏ *hai* vùng cùng lúc không? [Visual communication of object concepts at different levels of abstraction](https://escholarship.org/uc/item/26c295st)

## Cơ chế duy nhất

Thí nghiệm sử dụng bốn vùng kích thước bằng nhau trong lưới patch của mỗi ảnh. Trên lưới $7\times7$ của ViT-B/32, mỗi góc cung cấp chín patch, hàng giữa và cột giữa tạo thành phần ngữ cảnh luôn có mặt. Bỏ các token của vùng được can thiệp trước block transformer đầu tiên. Những token còn lại giữ tọa độ gốc. Không cần bản đồ saliency, mask vật thể hay thông tin từ ảnh gallery khác. Bốn vùng là bốn đơn vị thí nghiệm có thể tái lập. Chúng không được chú thích là bộ phận giải phẫu hay bộ phận của vật thể.

Với ảnh $X$ và tập vùng bị bỏ $S$, dùng CLIP đóng băng và đúng prompt của miền để thu phân bố từ $p_X^S$. Sketch được đọc ở ảnh gốc và ảnh lật, quy các mask về cùng vị trí vật lý trước khi lấy trung bình posterior. Đối với một sketch $X$ và một ảnh $Y$, đặt $c_{S,T}=\sum_w p_X^S(w)p_Y^T(w)$. Đây là xác suất hai lượt lấy mẫu tên độc lập từ hai phân bố trả cùng một từ.

Lấy một hoán vị $\pi$ của bốn vùng với prior đều. Gọi $E_1$ là biến cố cả bốn cặp thí nghiệm bỏ một vùng đều có tên trùng nhau. Gọi $E_2$ là biến cố tương tự trên cả sáu cặp thí nghiệm bỏ hai vùng. Các lần lấy mẫu tên độc lập có điều kiện theo ảnh và $\pi$. Định nghĩa

$$
A_\pi=\prod_{i=1}^4 c_{i,\pi(i)},\qquad
B_\pi=\prod_{1\le i<j\le4}c_{\{i,j\},\{\pi(i),\pi(j)\}},\qquad
K(X,Y)=P(E_2\mid E_1,X,Y)=\frac{\sum_\pi A_\pi B_\pi}{\sum_\pi A_\pi}.
$$

Xếp hạng theo $\log K/6$. Phân bố của hoán vị được suy ra từ bốn phép can thiệp đơn. Sáu đáp ứng kép dùng để kiểm tra những hoán vị đó. Tất cả là tính toán tất định trên 24 khả năng. Xác suất trên mô tả phép thí nghiệm lấy mẫu tên được định nghĩa ở đây. Nó chưa được hiệu chuẩn thành xác suất hai ảnh cùng lớp.

Phương pháp bất biến với cách đặt tên bốn vùng trong mỗi ảnh nếu đổi tên nhất quán giữa hai bậc can thiệp. Khi mọi can thiệp cho cùng posterior như ảnh gốc, $K^{1/6}$ bằng đúng Dot 601 từ. Hai cặp ảnh có cùng tập đáp ứng đơn và cùng tập đáp ứng kép vẫn có thể có điểm khác nhau nếu quan hệ giữa hai tập không giống nhau. Thay đổi cách gắn đáp ứng kép vào các vùng, trong khi giữ mọi vector, tạo một đối chứng cho sự khác biệt này. Các tích và tổng được tính trong miền log để tránh tràn dưới.

## Thiết kế khóa trước khi chấm

- Dùng OpenAI CLIP ViT-B/32, 601 từ Open Images đã duyệt, một prompt cho mỗi miền, seed 42, inference mode. Không dùng dữ liệu để học, không gradient. Mỗi vector mask có thể tính trước theo ảnh riêng lẻ. Một ảnh thêm hoặc bớt khỏi gallery không đổi điểm giữa hai ảnh khác.
- Đối chứng quan trọng nhất là đặt posterior của hoán vị đều trước khi dự đoán sáu đáp ứng kép, tức $K_0=\frac1{24}\sum_\pi B_\pi$. $K_0$ cũng là kỳ vọng **trên thang xác suất** khi ngẫu nhiên tráo danh tính của đáp ứng kép đích so với đáp ứng đơn bằng một hoán vị chung. Nó giữ nguyên tất cả pixel, các lần gọi CLIP, số vùng và cấu trúc tổ hợp. Với prior đều, $K-K_0=\operatorname{Cov}_\pi(A_\pi,B_\pi)/\mathbb E_\pi A_\pi$. Phần khác biệt đúng bằng khả năng đáp ứng đơn dự báo đáp ứng kép.
- So thêm với ánh xạ cố định theo tọa độ, với Dot trên posterior trung bình của các đáp ứng kép, và với đáp ứng kép tái dựng từ đáp ứng đơn. Trong đối chứng cuối, dự đoán $p_{ij}^{(0)}(w)\propto p_i(w)p_j(w)/p_{\emptyset}(w)$, là giả thuyết cộng tác động trên log-odds. Các đối chứng chỉ dùng để giải thích kết quả.
- Hai split Sketchy và TU-Berlin 30 lớp dùng full unseen, cùng evaluator KD-SBIR. Hai chỉ số là mAP@all / P@100 cho Ext-1 và TU-Berlin, mAP@200 / P@200 cho Ext-2. So cả tỷ lệ ảnh đúng lớp trong **cùng số lượng ảnh mới vào top-K** so với Dot. Bootstrap theo lớp. Đối chiếu ảnh nguồn thật trong lớp trên Sketchy là phân tích phụ.
- Giả thuyết chính cần một lợi ích theo lớp so với $K_0$ ở cùng ngân sách ảnh mới và chuyển được sang TU-Berlin với cấu hình khóa. Việc chỉ thắng Dot không chứng minh hai bậc can thiệp có sự tương ứng. Nếu $K_0$ giải thích hết lợi ích hoặc chỉ những cặp cùng một ảnh vật lý hưởng lợi, thu hẹp kết luận tương ứng. Không chỉnh hyperparameter trong lần thử đầu.

## Prior art và mức độ mới cần kiểm chứng

MAE cho encoder chỉ nhìn thấy token còn lại và học để phục dựng pixel. Ở đây chỉ kế thừa cách đưa token quan sát được qua ViT, dùng mọi trọng số CLIP sẵn có. [He et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html)

Nghiên cứu về giải thích mạng thị giác đã so sánh các tập patch đủ để giữ quyết định trên một ảnh. Việc đo đáp ứng theo tập patch đã có tiền lệ. Hạng $p_i p_j/p_{\emptyset}$ là đối chứng cụ thể cho phần đáp ứng tổ hợp có thể suy từ các can thiệp đơn. [Jiang et al., CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Jiang_Comparing_the_Decision-Making_Mechanisms_by_Transformers_and_CNNs_via_Explanation_CVPR_2024_paper.html)

Đối sánh xác suất giữa các đồ thị và siêu đồ thị đã dùng thông tin của nhiều điểm để suy ra một phép gán chung. Đối sánh bậc cao và lấy tổng trên các hoán vị có tiền lệ. Phần nghiên cứu cụ thể ở đây là thay đặc trưng hình học tĩnh bằng **kết quả của các thí nghiệm xóa token**, dùng một bậc can thiệp để suy ra tương ứng, rồi dùng xác suất dự báo trên bậc khác làm tiêu chí truy hồi. Việc diễn đạt lại dưới dạng xác suất chưa tự nó chứng minh novelty độc lập. [Zass & Shashua, CVPR 2008](https://doi.org/10.1109/CVPR.2008.4587500)

SpLIP học prompt và dùng tác vụ jigsaw có giám sát. FreeDom dịch ảnh truy vấn sang mô tả rời rạc qua các ảnh proxy. BASIC phối hợp bằng chứng ảnh và văn bản cho ảnh truy vấn có chỉnh sửa, còn STiTch dùng chuyển tiếp và vận chuyển ngữ nghĩa cho caption và ảnh. Các công trình đó không cung cấp trực tiếp phép kiểm chứng giữ lại một bậc can thiệp của từng cặp ảnh này. [SpLIP, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/3426_ECCV_2024_paper.php) · [FreeDom, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/papers/Efthymiadis_Composed_Image_Retrieval_for_Training-Free_Domain_Conversion_WACV_2025_paper.pdf) · [BASIC, NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/hash/6391a02f02dc529cfa3f242fb4343c6a-Abstract-Conference.html) · [STiTch, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Li_STiTch_Semantic_Transition_and_Transportation_in_Collaboration_for_Training-Free_Zero-Shot_CVPR_2026_paper.html)

## Cách nhận ra tín hiệu thuộc về sự tương ứng

Gọi $u_\pi=A_\pi/\sum A$ và $v_\pi=B_\pi/\sum B$. Từ định nghĩa trên suy ra $K/K_0=24\sum_\pi u_\pi v_\pi$. Tỷ số này là mức ủng hộ một phép gán dùng chung so với hai phép gán độc lập trong thí nghiệm lấy mẫu tên. Nếu một bậc can thiệp cho posterior đều trên 24 phép gán, tỷ số bằng 1 bất kể đáp ứng của bậc kia. Nhân ma trận tương hợp của từng bậc với một hằng số dương riêng cho từng hàng hoặc từng cột cũng không đổi tỷ số: mọi song ánh sử dụng mỗi hàng và mỗi cột đúng một lần. Các thuộc tính này cho phép kiểm tra đúng phần của phương pháp có khả năng mang thông tin mới.

Trong top-K của Dot, ghép các ảnh kề nhau theo $K_0$ trước khi xem nhãn. Trong những cặp có một ảnh đúng lớp, đo tỷ lệ ảnh đúng nhận log tỷ số cao hơn. Báo cả trung bình theo ảnh lẫn trung bình theo lớp, với mỗi truy vấn có trọng số ngang nhau trong lớp. Lặp phép đo sau khi loại hoán vị đồng nhất và chuẩn hóa lại trên 23 khả năng còn lại để xem kết luận có phụ thuộc hoàn toàn vào cùng bố cục ảnh hay không. Ngoài số đo tổng hợp, lưu ví dụ đầu tiên theo thứ tự dữ liệu ở hai trường hợp top-1 được sửa hoặc bị đổi thành sai.

Liang et al. đã đo việc CLIP đọc kém trên ảnh có mask trong segmentation. Ở đây token bị bỏ hoàn toàn và ngữ cảnh trung tâm luôn còn lại, nhưng mô hình vẫn có thể mất bằng chứng khi thiếu một vùng. Vì vậy mọi đối chứng định danh dùng cùng các lần đọc đã che. Việc dùng mask trên hai ảnh cũng có tiền lệ trong giải thích mô hình tương đồng, nơi người ta dự báo thuộc tính giải thích một điểm khớp có sẵn. Phép thử hiện tại dùng bậc can thiệp chưa sử dụng để trực tiếp dự báo một quan sát trong truy hồi. [Liang et al., CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Liang_Open-Vocabulary_Semantic_Segmentation_With_Mask-Adapted_CLIP_CVPR_2023_paper.html) · [Why Do These Match?, ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560630.pdf)
