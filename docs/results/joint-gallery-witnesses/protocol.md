# Suy luận từ một nhóm ảnh cùng hỗ trợ một phân bố từ

## Giả thuyết

Dot trên 601 từ rút gọn hồ sơ ngữ nghĩa của từng ảnh thành một số. Hai ảnh có thể đạt cùng điểm nhờ hai cách đọc khác nhau của truy vấn. Trong nghiên cứu trước, chỉ kiểm tra bao lồi trên điểm Dot chấp nhận các nhóm đó. Kiểm tra riêng từng từ hạn chế đổi nghĩa bên trong một từ, nhưng có thể lấy một nhóm ảnh cho mỗi từ rồi cộng nhiều bằng chứng chưa từng được cùng quan sát. Đối chứng trước đó giúp xác nhận vai trò hướng hình học ở cùng số ảnh mới. Suy luận từng từ và sau Dot đạt tỷ lệ đúng lớp gần ngang nhau. [Thực nghiệm dẫn đến giả thuyết](../gallery-convex-support/README.md)

Phép thử này yêu cầu tồn tại **một** nhóm ảnh và **một** phần phân bố ngữ nghĩa mà mọi ảnh trong nhóm cùng hỗ trợ. Nhóm ảnh phải bao quanh ứng viên trên mặt phẳng tiếp tuyến cục bộ. Dự đoán cần kiểm chứng là, với cùng số ảnh mới vào top-K, việc giữ điều kiện chung làm giảm nhầm lẫn giữa các cách đọc và tăng tỷ lệ ảnh đúng lớp. Mức tăng mAP so với baseline là số đo bổ sung. Một khác biệt đại số trong dữ liệu dựng chưa đủ xác nhận giả thuyết trên ảnh thật.

## Định nghĩa

\(q\) là posterior 601 từ của sketch, \(p_i\) là posterior của ảnh \(i\), và \(s_i=q^\top p_i\). Giữ 16 láng giềng CLIP RGB gần nhất và mặt phẳng 2D của PCA đã trừ trung bình phần dư tiếp tuyến. Tọa độ \(z_{ij}\) đo từ \(i\), giữ ứng viên tại gốc. Tập nhân chứng khả thi là

\[
\mathcal W_i=\left\{J\subseteq N_i:0\in \operatorname{int}\operatorname{conv}
       \left(\{z_{ij}:j\in J\}\right)\right\}.
\]

Với mỗi nhóm \(J\), phần phân bố chung lớn nhất theo từng tọa độ là

\[
u^{(J)}_w=\min_{j\in J} p_{jw}.
\]

Mỗi \(u^{(J)}\) là một subdistribution, tức tổng khối lượng có thể dưới 1. Chính khối lượng đó giảm khi các thành viên chỉ hỗ trợ những từ khác nhau. Điểm của ứng viên là

\[
S_i^{\mathrm{joint}}(q)=\max\left\{s_i,\ \max_{J\in\mathcal W_i} q^\top u^{(J)}\right\},
\qquad \max\varnothing=0.
\]

Tương đương, tối đa hóa \(q^\top u\) trên các \(u\ge0\) mà tập ảnh thoả \(p_j\succeq u\) bao quanh ứng viên. Tập vector khả thi là hợp của các hộp \([0,u^{(J)}]\). Chọn trước một nhóm riêng cho mỗi từ sẽ thay hợp này bằng một hộp lớn hơn và có thể tạo ra tổ hợp ngữ nghĩa chưa được xác nhận chung.

Trong không gian embedding affine, \(\log p_w(x)\) là logit tuyến tính trừ log-sum-exp, nên log-concave. Các điều kiện đồng thời \(p_w(x)\ge u_w\) là giao của các tập lồi. Bởi vậy nếu từng nhân chứng nằm trong tập xác nhận chung và ứng viên là nội suy trong cùng không gian, ứng viên cũng nằm trong tập đó. Quy tắc thực dùng chart PCA cục bộ trên mặt cầu chuẩn hóa. Sự bảo toàn điều kiện sau chiếu là một giả định thực nghiệm, có thể hỏng tại giao nhau giữa các chế độ ảnh hoặc khi CLIP đổi từ đồng nghĩa.

### Các đối chứng có cùng hình học

Với \(J\) đã cố định, \(q^\top\min_{j\in J}p_j\le \min_{j\in J}q^\top p_j\). Lấy cực đại theo \(J\) giữ bất đẳng thức. Nhánh “sau Dot” là chính vế phải với cùng \(\mathcal W_i\). Nhánh “từng từ” lấy cực đại theo nhóm riêng cho từng tọa độ, nên cũng là cận trên của phần bằng chứng chung. Phép suy luận giữ điều kiện nhóm, đo độ tương thích bằng cùng Dot 601 từ.

Ví dụ sáu ảnh cách đều theo góc. Mỗi ảnh có khối lượng .89 cho một trong hai từ, .01 cho từ kia, .10 cho phần còn lại. \(q=(.5,.5,0)\) và \(p_i=(.01,.01,.98)\). Mọi ảnh nhân chứng có Dot .45. Khi các ảnh hỗ trợ một từ cùng nằm về một phía, phần phân bố chung của mọi nhóm bao quanh chỉ cho .01. Khi hai từ xen kẽ quanh gốc, mỗi từ có một tam giác riêng. Nhánh từng từ cho .89 còn chứng nhận chung cho .45. Phép dựng bảo toàn Dot và vị trí nhân chứng, đồng thời phân biệt được hai kiểu ghép bằng chứng. Bản chạy kiểm tra các đẳng thức này và cận trên trên điểm thực.

### Prior art và chi phí

- PCA tiếp tuyến ước lượng biến thiên cục bộ từ gallery. Histogram intersection dùng cực tiểu từng tọa độ để đo phần chung. Ở đây đối tượng suy luận là phần posterior chung của một nhóm khả thi trong không gian ảnh. Cơ chế chọn nhóm theo truy vấn giữ cả điều kiện nội suy lẫn sự đồng thời của nhiều từ. [Zhang và Zha, LTSA](https://doi.org/10.1137/S1064827502419154) · [Swain và Ballard, Color Indexing, IJCV 1991](https://www.cs.utexas.edu/~dana/Swain1.pdf)
- Manifold diffusion truyền điểm qua các cạnh. ICFRR cập nhật thứ hạng bằng quan hệ gallery–gallery. STiTch đối sánh tập thành phần giữa truy vấn và từng ảnh. Phép suy luận này hỏi một câu hỏi khả thi trên một tập ảnh và có thể phân biệt hai gallery mà mọi điểm Dot giống hệt. [Iscen et al., CVPR 2017](https://arxiv.org/abs/1611.05113) · [Hudson và Smith, ICFRR](https://arxiv.org/abs/2303.17703) · [Li et al., STiTch, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Li_STiTch_Semantic_Transition_and_Transportation_in_Collaboration_for_Training-Free_Zero-Shot_CVPR_2026_paper.html)
- Chỉ cần liệt kê nhóm bao quanh tối tiểu, vì thêm nhân chứng vào một nhóm làm \(u^{(J)}\) giảm hoặc giữ nguyên. Trong 2D cần tối đa bốn điểm, bao gồm trường hợp nằm trên giao của hai đoạn. Với ngưỡng số học \(\delta=10^{-6}\), mỗi nhân chứng phủ một cung hướng dài \(L=\pi-2\arcsin\delta\). Trong nhóm tối tiểu, tổng hai khoảng góc kề mỗi nhân chứng ít nhất \(L\), vì khi nhỏ hơn thì hai cung bên cạnh phủ cung ở giữa. Cộng qua cả nhóm cho \(mL\le4\pi\), nên \(m\le4\). Mã kiểm tra từng nhóm ba, thêm nhóm bốn chỉ nếu mọi nhóm con ba đều không đủ. Đối chiếu với vét cạn \(2^{16}\) tập trên 32 chart mỗi split. [Định lý Steinitz và giới hạn 2d](https://doi.org/10.1112/blms.12965)

Các vector chung được dựng theo block ứng viên. GPU dùng nhân ma trận trên đủ 601 tọa độ và lấy cực đại theo nhóm của mỗi ảnh. Với \(M\) nhóm bao quanh tối tiểu, chi phí chấm là \(O(601M)\) mỗi sketch và cao hơn baseline \(O(601N)\). Block giữ bộ nhớ theo kích thước phần gallery đang tính. \(S_i^{\mathrm{joint}}\) tăng điểm đến tối đa mức nhân chứng ban đầu và giữ nguyên top 1 khi dùng Dot gốc phá hòa.

## Thực nghiệm khóa trước

Chấm trên toàn bộ unseen của Sketchy Ext-1, Ext-2 và TU-Berlin 30 lớp, với đủ 14.179 / 12.694 / 2.130 truy vấn và 17.100 / 12.552 / 27.262 ảnh. OpenAI CLIP ViT-B/32 đóng băng, cùng prompt, đầy đủ 601 từ Open Images boxable, seed 42, gallery không nhãn, mỗi sketch độc lập. Mọi điểm mới chấm bằng KD-SBIR gốc. Ext-1 và TU-Berlin dùng mAP@all/P@100, Ext-2 dùng mAP@200/P@200. Đối chứng từ nghiên cứu trước dùng cùng checkpoint và graph. ID, SHA của dữ liệu và precision từng truy vấn được kiểm tra lại.

Phép thử chính ghép cùng số ảnh mới trong top-K theo từng truy vấn. Với nhánh chính và mỗi đối chứng, lấy \(\min(n_{\mathrm{joint}},n_{\mathrm{control}})\) ảnh mới xếp cao nhất, sau đó mới đối chiếu nhãn. Báo cáo bootstrap theo lớp và tỷ lệ ảnh đúng tại các lần chỉ nhánh sau Dot vượt ngưỡng điểm của hạng K gốc. Hai split Sketchy và TU-Berlin dùng cùng công thức và tham số. Phạm vi thực nghiệm kiểm tra chuyển giữa benchmark trong cùng cặp miền.

~~~sh
PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -u -m tf_sbir.joint_gallery_witnesses
~~~
