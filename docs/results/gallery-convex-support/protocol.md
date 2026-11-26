# Chứng nhận cùng một khái niệm từ các hướng lân cận trong gallery

## Giả thuyết

Dot đủ 601 từ có dạng \(s_i(q)=\sum_w q_w p_{iw}\), với \(q\) là posterior từ của truy vấn và \(p_i\) là posterior từ của ảnh \(i\). Một ảnh lân cận có thể đạt Dot cao nhờ “chim”, ảnh lân cận phía khác đạt cùng điểm nhờ “máy bay”. Nếu chỉ giữ \(s\), cả hai ảnh có thể cùng truyền bằng chứng đến một ứng viên mà không có cách đọc nào của truy vấn được xác nhận từ các hướng khác nhau. Số ảnh lân cận có điểm cao cũng có thể lớn khi chúng đều lặp lại cùng một góc nhìn.

Giả thuyết cần kiểm chứng là việc giữ danh tính từ cho đến sau suy luận trên gallery giúp phân biệt hai tình huống đó. Một khái niệm có bằng chứng nội suy mạnh hơn nếu, với mọi đường qua ứng viên trong một tọa độ hình học cục bộ, vẫn có ảnh giữ mức bằng chứng ấy ở cả hai phía. Phép thử dùng cùng 601 từ cho mỗi truy vấn, mỗi ảnh và mỗi đối chứng.

### Vì sao phép cộng Dot có thể làm mất điều kiện nội suy

Trong không gian embedding tuyến tính trước phép chuẩn hóa lại trên một đường nội suy, CLIP dùng \(p_w(x)=\exp(\beta t_w^\top x)/\sum_v\exp(\beta t_v^\top x)\). Hessian của \(\log p_w(x)\) bằng \(-\beta^2\operatorname{Cov}_{v\sim p(x)}(t_v)\), là bán xác định âm. Do đó tập mức trên của **một từ** lồi trong không gian tuyến tính này. Điểm Dot là hỗn hợp những xác suất ấy. Điều kiện lồi của tập mức có thể mất sau khi lấy hỗn hợp. Chẳng hạn logits \((-a,0,a)\) và \(q=(1/2,0,1/2)\) cho \(s(a)=\cosh(a)/(1+2\cosh(a))\). Điểm ở hai đầu \(-a,+a\) cao hơn điểm ở giữa dù mỗi xác suất từ riêng vẫn log-concave.

Đây là một giới hạn của việc dùng một điểm gộp làm dữ liệu cho suy luận, kể cả khi các logit cục bộ thật sự affine. Tính log-concave này áp dụng trên đoạn thẳng embedding. Ảnh quan sát có embedding chuẩn hóa trên mặt cầu, còn công thức dùng bao lồi trong chart hai chiều, nên điều kiện nội suy của bằng chứng thật trong chart vẫn cần được kiểm chứng. Phép suy ra Hessian làm rõ giả thuyết và giữ nguyên công thức, cấu hình và thiết kế đã khóa ở commit chạy.

## Suy luận

Chuẩn hóa embedding RGB của CLIP ViT-B/32. Với mỗi ảnh \(i\), lấy \(N_i\) gồm 16 láng giềng gần nhất theo cosine, bỏ chính nó. Đặt \(a_{ij}=x_j-(x_i^\top x_j)x_i\). Ước lượng hai trục \(U_i\) bằng PCA trên \(a_{ij}-\bar a_i\). Khi đo vị trí, dùng \(z_{ij}=U_i^\top a_{ij}\), giữ ứng viên ở gốc. Chart tham khảo xấp xỉ tiếp tuyến cục bộ của Zhang và Zha. Hai chiều cho phép kiểm tra bao quanh bằng một phép quét góc chính xác. Cùng 16 láng giềng được dùng trong đối chứng diffusion. [Zhang và Zha, SIAM Journal on Scientific Computing](https://doi.org/10.1137/S1064827502419154)

Lượt đầu ở commit 5641e33 ước lượng hai trục bằng PCA qua gốc trên chính \(a_{ij}\). Nếu Gram \(A_i A_i^\top\) dương từng phần tử, định lý Perron–Frobenius cho biết thành phần đầu có cùng dấu trên mọi láng giềng. Khi đó hai chiều được chọn luôn để mọi nhân chứng về một phía, và chứng nhận bất khả thi ngay cả khi điểm mọi nhân chứng bằng 1. Kiểm tra trên gallery của hai split Sketchy cho thấy trường hợp Gram dương phổ biến. Trừ trung bình lúc ước lượng trục loại hạng \(\bar a_i\bar a_i^\top\) khỏi moment bậc hai. Ảnh ứng viên vẫn ở gốc khi kiểm tra nội suy. Lượt sửa giữ nguyên cơ chế chứng nhận, số láng giềng, số chiều và đối chứng. Quyết định sửa dựa trên tính khả thi hình học của gallery Sketchy và được khóa trước lượt chấm mới.

Với một trường điểm không âm \(v\) trên gallery, đặt

\[
H_i(v)=\sup\left(\{0\}\,\cup\,\{t>0:0\in\operatorname{int}
 \operatorname{conv}\{z_{ij}:j\in N_i,\ v_j\ge t\}\}\right).
\]

Theo định lý siêu phẳng tách, \(H_i(v)=\min_{\|a\|=1}\max_{j\in N_i:\ a^\top z_{ij}>0} v_j\), với giá trị cực đại của tập rỗng bằng 0. Vì vậy có thể đọc \(H_i\) là mức bằng chứng vẫn còn sau khi loại một nửa mặt phẳng theo bất kỳ hướng nào. Phép thử dùng nội điểm trong hai chiều để yêu cầu bằng chứng ở mọi hướng. Trong một chart cố định, lặp lại một nhân chứng tại cùng vị trí với cùng điểm giữ nguyên tập bao lồi. Việc thay gallery có thể thay chart và tập láng giềng.

Với một tập mức nhân chứng hữu hạn, điểm 0 ở nội điểm khi và chỉ khi tồn tại một bán kính dương để toàn bộ đĩa quanh 0 nằm trong bao lồi. Vì vậy \(H_i\) là mức nội suy cao nhất còn đúng dưới một nhiễu vị trí đủ nhỏ của điểm cần chấm, giữ chart và nhân chứng cố định. Mức này bất biến dưới phép đổi cơ sở khả nghịch trong chart.

Nếu tập mức trên của độ liên quan thật cho từ \(w\) lồi cục bộ trong chart và \(p_{jw}\ge t\) đủ để xác nhận láng giềng \(j\) ở trong tập mức \(t\), thì \(H_i(p_{\cdot w})\) là một mức được suy ra cho \(i\). Hai điều kiện là giả thuyết về dữ liệu. Chúng có thể hỏng khi chart gộp các nhánh của manifold hoặc nhiều phía đều chứa cùng một nhầm lẫn CLIP. Bao lồi của tập mức và phép bao tựa lõm có tiền lệ toán học. [Cuoghi và Salani, Electronic Journal of Differential Equations 2006, công thức (3.1)](https://ejde.math.txstate.edu/Volumes/2006/124/cuoghi.pdf)

Điểm suy luận là

\[
r_{iw}=\max\{p_{iw},H_i(p_{\cdot w})\},\qquad
\tilde{s}_i(q)=\mathbb E_{W\sim q}[r_{iW}]=\sum_{w=1}^{601}q_w r_{iw}.
\]

Với \(W\) cố định và mức \(t\) lấy đều trên \([0,1]\), \(r_{iW}\) là xác suất ảnh được chấp nhận trực tiếp hoặc có bằng chứng nội suy ở mức \(t\). Các tọa độ \(r_{iw}\) biểu diễn các sự kiện chấp nhận riêng, tổng của chúng có thể lớn hơn 1. Gallery có thể lưu \(r\) từ trước và chấm online bằng một tích ma trận cùng kích thước baseline. Trong hai chiều, mỗi chart có tối đa 32 nửa mặt phẳng tới hạn. Mã khóa chúng thành mặt nạ bit, xếp điểm nhân chứng giảm dần rồi lấy mức đầu tiên phủ đủ mọi nửa mặt phẳng.

Khi \(q\) dồn hết xác suất vào một từ, kết quả trùng với việc áp dụng phép bao cho Dot. Tổng quát, đặt \(\varepsilon=1-\max_w q_w\) và \(R_i(v)=\max(v_i,H_i(v))\). Có thể đưa \(v_i\) vào mọi phép cực đại theo hướng, nên \(R_i\) đơn điệu, thuần nhất dương và giao hoán với cộng hằng số. Vì \(0\le p_{jw}\le 1\), cả \(\mathbb E_q[R_i(p_{\cdot W})]\) và \(R_i(\mathbb E_q[p_{\cdot W}])\) đều nằm trong đoạn \([(1-\varepsilon)R_i(p_{\cdot w_*}),(1-\varepsilon)R_i(p_{\cdot w_*})+\varepsilon]\) cho \(w_*=\arg\max_w q_w\). Do đó chênh điểm tuyệt đối tối đa là \(\varepsilon\). Điều này đặt giới hạn cho tác động của thứ tự suy luận trên truy vấn CLIP gần như chắc chắn về một từ. Thứ hạng vẫn có thể đổi nếu hai ảnh có điểm sát nhau.

## Điều khác biệt cần chứng minh

Xét sáu nhân chứng quanh một ứng viên với các góc cách nhau 60 độ. Truy vấn đặt cùng xác suất cho hai từ. Trong gallery A, cả sáu nhân chứng có xác suất .89 cho từ thứ nhất và .01 cho từ thứ hai. Trong gallery B, ba nhân chứng ở một phía hoán đổi hai xác suất đó. Mọi Dot trong gallery vẫn giống nhau. Với ứng viên có xác suất .01 cho mỗi từ, điểm mới là .45 ở A và .01 ở B. Bao lồi trên điểm Dot chấp nhận .45 trong cả hai. Phép dựng cho thấy bản đồ trước khi lấy kỳ vọng không khôi phục được từ riêng vector các điểm Dot, kể cả khi biết graph. Fixture chạy kiểm tra đẳng thức này trước lúc mở kết quả thực.

- Iscen et al. dùng truyền tín hiệu qua graph cố định để nắm cấu trúc manifold. Với cùng graph, truyền tuyến tính trước và sau tổng theo từ cho cùng kết quả. ICFRR dùng các thứ hạng gallery–gallery để cập nhật lặp thứ hạng cross-domain. ClusterRetri truy hồi qua đại diện cụm. Phép thử ở đây đặt một điều kiện nội suy cục bộ trên từng giả thuyết từ. [Iscen et al., CVPR 2017](https://arxiv.org/abs/1611.05113) · [Hudson và Smith, ICFRR](https://arxiv.org/abs/2303.17703) · [Wu et al., ClusterRetri](https://arxiv.org/abs/2301.06685)
- Truyền trên hypergraph cũng giữ được nhiều loại bằng chứng. Phương pháp NeurIPS 2021 của An et al. mở hyperedge qua đối sánh vị trí của đặc trưng cục bộ và homography. Điều kiện ở đây nằm trên hình học của nhiều ảnh gallery quanh một ứng viên và kiểm tra một từ được giữ xuyên suốt. [Hypergraph Propagation and Community Selection for Objects Retrieval](https://proceedings.neurips.cc/paper/2021/file/1da546f25222c1ee710cf7e2f7a3ff0c-Paper.pdf)
- FreeDom ánh xạ ảnh truy vấn qua từ để chuyển miền. BASIC tách visual similarity và text similarity rồi kết hợp điểm trong CIR. STiTch dùng đối sánh hai tập thành phần của truy vấn và một ảnh. Cơ chế đề xuất giữ biểu diễn truy vấn đã chốt và kiểm tra mỗi cách đọc từ tại ứng viên qua cấu trúc của nhiều ảnh gallery. [Efthymiadis et al., FreeDom, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/papers/Efthymiadis_Composed_Image_Retrieval_for_Training-Free_Domain_Conversion_WACV_2025_paper.pdf) · [Psomas et al., BASIC, NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/hash/6391a02f02dc529cfa3f242fb4343c6a-Abstract-Conference.html) · [Li et al., STiTch, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Li_STiTch_Semantic_Transition_and_Transportation_in_Collaboration_for_Training-Free_Zero-Shot_CVPR_2026_paper.html)
- Bressan et al. dùng tính lồi của lớp và truy vấn nhãn trong active learning. Agarwal et al. tính xác suất bao lồi khi sự xuất hiện hoặc vị trí của từng điểm là ngẫu nhiên. Ở đây mọi ảnh đã quan sát và giả thuyết từ được lấy chung từ posterior của truy vấn. Tính khả thi của bao lồi được kiểm tra có điều kiện theo một từ trước khi lấy kỳ vọng. [Margin-Based Active Learning of Multiclass Classifiers, JMLR 2024](https://www.jmlr.org/papers/volume25/22-1127/22-1127.pdf) · [Convex Hulls under Uncertainty](https://arxiv.org/abs/1406.6599)

Phần mượn trực tiếp là chart PCA cục bộ và toán của bao tựa lõm. Đóng góp đang được kiểm tra là thứ tự suy luận đối với khái niệm tiềm ẩn, cùng một đối chứng có thể giữ nguyên toàn bộ điểm Dot và chỉ thay cấu trúc xác nhận từng từ. Tính hữu ích của sự khác biệt cần kết quả trên lỗi thật.

## Thiết kế thực nghiệm khóa trước

1. **Dot** dùng đúng hai prompt và đủ 601 tọa độ gốc. Đối chiếu thứ tự ID và số đo với kết quả đã lưu theo từng truy vấn. **RGB diffusion** dùng cùng 16 láng giềng có hướng, trọng số cosine mũ ba chuẩn hóa theo hàng, restart \(\alpha=0.5\), 16 vòng. Tính trước trên 601 tọa độ, tương đương khuếch tán điểm Dot.
2. **Phép thử thứ tự** dùng \(\max(s_i,H_i(s))\). Khi bằng nhau, dùng Dot gốc để phá hòa. **Phép thử hình học** tráo hướng giữa các láng giềng trong từng nhóm bốn hạng cosine kề nhau bằng RNG seed 42. Tráo giữ đúng điểm mỗi ảnh, tập láng giềng, đa tập điểm mỗi từ và đa tập hướng trong mỗi nhóm. Điều bị thay là ghép ảnh với hướng.
3. Chấm mAP và precision bằng hàm KD-SBIR gốc với mọi truy vấn unseen. Đối chiếu phép tính trên GPU và hàm gốc cho từng batch. Ext-1 có 14.179 query, 17.100 ảnh, 25 lớp. Ext-2 có 12.694 query, 12.552 ảnh, 21 lớp. TU-Berlin có 2.130 query, 27.262 ảnh, 30 lớp. Ext-1 và TU-Berlin dùng mAP@all/P@100, Ext-2 dùng mAP@200/P@200. Cùng cấu hình cố định cho cả ba split.
4. Đếm sửa sai và gây sai ở top 1 so với Dot. Với ứng viên có Dot dưới điểm của hạng 100 hoặc 200 trước suy luận, đo tỷ lệ cùng lớp trong các trường hợp chỉ một trong hai nhánh vượt được mức điểm đó. Đây là chẩn đoán mức điểm, khác với tư cách thành viên của top-K mới. So sánh cặp theo lớp bằng bootstrap seed 42. Nhãn chỉ đi vào các phép chấm này, sau khi các nhánh đã tính xong điểm.

Cơ chế cần cho thấy sự xác nhận đúng hơn với hướng thật và với suy luận từng từ. Nếu một đối chứng đạt hiệu quả tương đương, mAP tăng trên Dot vẫn chỉ xác nhận giá trị của việc dùng gallery trong thiết lập này. Sketchy và TU-Berlin cùng đo sketch → photo. Kết quả có thể kiểm tra chuyển bộ dữ liệu trong cặp miền này. Công thức dùng một posterior truy vấn và đặc trưng gallery chung cho mọi miền ảnh có những tín hiệu đó.

Chạy trên Colab, dùng OpenAI CLIP ViT-B/32 đóng băng, một truy vấn mỗi lần suy luận về mặt dữ liệu, seed 42, cache encoder có sẵn và đúng KD-SBIR. Lệnh tái lập

~~~sh
PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -u -m tf_sbir.gallery_convex_support
~~~
