# Chứng nhận cùng một khái niệm từ các hướng lân cận trong gallery

## Giả thuyết

Dot đủ 601 từ có dạng \(s_i(q)=\sum_w q_w p_{iw}\), với \(q\) là posterior từ của truy vấn và \(p_i\) là posterior từ của ảnh \(i\). Một ảnh lân cận có thể đạt Dot cao nhờ “chim”, ảnh lân cận phía khác đạt cùng điểm nhờ “máy bay”. Nếu chỉ giữ \(s\), cả hai ảnh có thể cùng truyền bằng chứng đến một ứng viên mà không có cách đọc nào của truy vấn được xác nhận từ các hướng khác nhau. Số ảnh lân cận có điểm cao cũng có thể lớn khi chúng đều lặp lại cùng một góc nhìn.

Giả thuyết cần kiểm chứng là việc giữ danh tính từ cho đến sau suy luận trên gallery giúp phân biệt hai tình huống đó. Một khái niệm có bằng chứng nội suy mạnh hơn nếu, với mọi đường qua ứng viên trong một tọa độ hình học cục bộ, vẫn có ảnh giữ mức bằng chứng ấy ở cả hai phía. Phép thử dùng cùng 601 từ cho mỗi truy vấn, mỗi ảnh và mỗi đối chứng.

## Suy luận

Chuẩn hóa embedding RGB của CLIP ViT-B/32. Với mỗi ảnh \(i\), lấy \(N_i\) gồm 16 láng giềng gần nhất theo cosine, bỏ chính nó. Chiếu sai khác sang tiếp diện của mặt cầu đơn vị tại \(x_i\), rồi lấy hai phương PCA của các sai khác qua gốc. Gọi \(z_{ij}\) là tọa độ hai chiều của ảnh \(j\) trong chart tại \(i\). Chart tham khảo xấp xỉ tiếp tuyến cục bộ của Zhang và Zha. Giữ hai chiều để kiểm tra bao quanh bằng một phép quét góc chính xác và dùng cùng 16 láng giềng trong đối chứng diffusion. Hai lựa chọn này được khóa trước khi chấm. [Zhang và Zha, SIAM Journal on Scientific Computing](https://doi.org/10.1137/S1064827502419154)

Với một trường điểm không âm \(v\) trên gallery, đặt

\[
H_i(v)=\sup\left(\{0\}\,\cup\,\{t>0:0\in\operatorname{int}
 \operatorname{conv}\{z_{ij}:j\in N_i,\ v_j\ge t\}\}\right).
\]

Theo định lý siêu phẳng tách, \(H_i(v)=\min_{\|a\|=1}\max_{j\in N_i:\ a^\top z_{ij}>0} v_j\), với giá trị cực đại của tập rỗng bằng 0. Vì vậy có thể đọc \(H_i\) là mức bằng chứng vẫn còn sau khi loại một nửa mặt phẳng theo bất kỳ hướng nào. Phép thử dùng nội điểm trong hai chiều để yêu cầu bằng chứng ở mọi hướng. Trong một chart cố định, lặp lại một nhân chứng tại cùng vị trí với cùng điểm giữ nguyên tập bao lồi. Việc thay gallery có thể thay chart và tập láng giềng.

Nếu tập mức trên của độ liên quan thật cho từ \(w\) lồi cục bộ trong chart và \(p_{jw}\ge t\) đủ để xác nhận láng giềng \(j\) ở trong tập mức \(t\), thì \(H_i(p_{\cdot w})\) là một mức được suy ra cho \(i\). Hai điều kiện là giả thuyết về dữ liệu. Chúng có thể hỏng khi chart gộp các nhánh của manifold hoặc nhiều phía đều chứa cùng một nhầm lẫn CLIP. Bao lồi của tập mức và phép bao tựa lõm có tiền lệ toán học. [Cuoghi và Salani, Electronic Journal of Differential Equations 2006, công thức (3.1)](https://ejde.math.txstate.edu/Volumes/2006/124/cuoghi.pdf)

Điểm suy luận là

\[
r_{iw}=\max\{p_{iw},H_i(p_{\cdot w})\},\qquad
\tilde{s}_i(q)=\mathbb E_{W\sim q}[r_{iW}]=\sum_{w=1}^{601}q_w r_{iw}.
\]

Với \(W\) cố định và mức \(t\) lấy đều trên \([0,1]\), \(r_{iW}\) là xác suất ảnh được chấp nhận trực tiếp hoặc có bằng chứng nội suy ở mức \(t\). Các tọa độ \(r_{iw}\) biểu diễn các sự kiện chấp nhận riêng, tổng của chúng có thể lớn hơn 1. Gallery có thể lưu \(r\) từ trước và chấm online bằng một tích ma trận cùng kích thước baseline. Trong hai chiều, mỗi chart có tối đa 32 nửa mặt phẳng tới hạn. Mã khóa chúng thành mặt nạ bit, xếp điểm nhân chứng giảm dần rồi lấy mức đầu tiên phủ đủ mọi nửa mặt phẳng.

## Điều khác biệt cần chứng minh

Xét sáu nhân chứng quanh một ứng viên với các góc cách nhau 60 độ. Truy vấn đặt cùng xác suất cho hai từ. Trong gallery A, cả sáu nhân chứng có xác suất .89 cho từ thứ nhất và .01 cho từ thứ hai. Trong gallery B, ba nhân chứng ở một phía hoán đổi hai xác suất đó. Mọi Dot trong gallery vẫn giống nhau. Với ứng viên có xác suất .01 cho mỗi từ, điểm mới là .45 ở A và .01 ở B. Bao lồi trên điểm Dot chấp nhận .45 trong cả hai. Phép dựng cho thấy bản đồ trước khi lấy kỳ vọng không khôi phục được từ riêng vector các điểm Dot, kể cả khi biết graph. Fixture chạy kiểm tra đẳng thức này trước lúc mở kết quả thực.

- Iscen et al. dùng truyền tín hiệu qua graph cố định để nắm cấu trúc manifold. Với cùng graph, truyền tuyến tính trước và sau tổng theo từ cho cùng kết quả. ICFRR dùng các thứ hạng gallery–gallery để cập nhật lặp thứ hạng cross-domain. ClusterRetri truy hồi qua đại diện cụm. Phép thử ở đây đặt một điều kiện nội suy cục bộ trên từng giả thuyết từ. [Iscen et al., CVPR 2017](https://arxiv.org/abs/1611.05113) · [Hudson và Smith, ICFRR](https://arxiv.org/abs/2303.17703) · [Wu et al., ClusterRetri](https://arxiv.org/abs/2301.06685)
- Truyền trên hypergraph cũng giữ được nhiều loại bằng chứng. Phương pháp NeurIPS 2021 của An et al. mở hyperedge qua đối sánh vị trí của đặc trưng cục bộ và homography. Điều kiện ở đây nằm trên hình học của nhiều ảnh gallery quanh một ứng viên và kiểm tra một từ được giữ xuyên suốt. [Hypergraph Propagation and Community Selection for Objects Retrieval](https://proceedings.neurips.cc/paper/2021/file/1da546f25222c1ee710cf7e2f7a3ff0c-Paper.pdf)
- FreeDom ánh xạ ảnh truy vấn qua từ để chuyển miền. BASIC tách visual similarity và text similarity rồi kết hợp điểm trong CIR. Cơ chế đề xuất giữ biểu diễn truy vấn đã chốt và kiểm tra mỗi cách đọc từ tại ứng viên qua cấu trúc của gallery. [Efthymiadis et al., FreeDom, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/papers/Efthymiadis_Composed_Image_Retrieval_for_Training-Free_Domain_Conversion_WACV_2025_paper.pdf) · [Psomas et al., BASIC, NeurIPS 2025](https://arxiv.org/abs/2510.25387)

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
