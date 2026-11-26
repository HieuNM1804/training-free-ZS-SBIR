# Hồi phục thành phần gallery được cả hai lần đọc truy vấn xác nhận

## Giả thuyết

Dot đủ 601 từ của baseline dùng trung bình hai posterior CLIP của cùng một sketch, ảnh gốc và ảnh lật ngang. Mỗi ảnh gallery chỉ nhận một điểm. Cách đọc rút gọn này mất thông tin lần quan sát nào tạo ra điểm cao. Hai lần đọc có thể ưu tiên hai ảnh khác nhau vì chúng thể hiện các biến thiên của cùng một đối tượng. Chúng cũng có thể ưu tiên hai họ ảnh tách biệt vì một lần đọc nhầm nội dung. Giả thuyết là **vị trí của việc đổi lần đọc trên gallery** giúp phân biệt hai trường hợp.

Phương pháp xử lý hai kết quả đo của một truy vấn. Độ tương quan giữa hai kết quả được giữ nguyên trong protocol và không có giả định độc lập thống kê. Khi cả hai cho cùng một vector điểm, phương pháp trả đúng thứ hạng Dot. Công thức chỉ dùng các lần đọc của cùng một truy vấn và quan hệ hình học trong gallery, nên cũng áp dụng được khi hai kết quả đo đến từ những phép biến đổi giữ nội dung ở miền ảnh khác.

## Cơ chế

Gọi \(a_i=(q^{\mathrm{gốc}})^\top p_i\), \(b_i=(q^{\mathrm{lật}})^\top p_i\) với đủ 601 từ. \(G\) là graph vô hướng của 16 láng giềng cosine CLIP RGB có quan hệ hai chiều. Đặt \(u_i=\max(a_i,b_i)\). Ở mức \(t\), thành phần của ảnh \(i\) trong graph con \(G[\{j:u_j\ge t\}]\) hợp lệ nếu chứa một ảnh có \(a_j\ge t\) và một ảnh có \(b_\ell\ge t\). Hai ảnh có thể khác nhau. Điểm ảnh \(i\) là mức \(t\) lớn nhất còn hợp lệ.

Mỗi thành phần hợp lệ có một đỉnh chung hoặc một cạnh đổi lần đọc. Thật vậy, nếu một thành phần chứa cả hai loại bằng chứng và không có đỉnh chứa cả hai, đường nối hai loại phải đi qua một cạnh có hai đầu thuộc hai loại. Vì vậy, ta tính các hạt giống \(\sigma\) bằng

\[
\sigma_i=\max_{j\in N_i\cup\{i\}}
             \max\{\min(a_i,b_j),\min(b_i,a_j)\}.
\]

Hồi phục hình thái học của \(\sigma\) dưới trần \(u\) tìm đúng các thành phần hợp lệ ở mọi mức điểm. Tương đương, nghiệm là

\[
c_i=\max_{\pi:i\leadsto v}\min\left(\sigma_v,\min_{j\in\pi}u_j\right).
\]

Khởi tạo \(c^{(0)}=\sigma\), lấy cực đại của trạng thái trong láng giềng gồm chính \(i\), sau đó chặn ở \(u_i\). Vòng lặp tăng đơn điệu và đạt điểm cố định sau tối đa \(N-1\) lần mở rộng đường đi. Đường chứa chu trình có thể bỏ chu trình mà không giảm mức nhỏ nhất. Xếp ảnh theo \(c_i\), phá hòa bằng Dot gốc. Nếu chỉ xét hạt giống \(d_i=\min(a_i,b_i)\), phép hồi phục yêu cầu có ít nhất một ảnh cho đồng thời hai điểm cao trong thành phần. Đây là đối chứng cho vai trò riêng của cạnh đổi lần đọc.

## Phân biệt với suy luận từ một điểm gộp

Trên graph đường \(1-2-3\), cho \(a=(.9,.8,.6)\) và \(b=(.1,.2,.4)\). Mỗi ảnh có Dot trung bình .5. Mọi thành phần chỉ có bằng chứng lần đọc thứ nhất trên mức .4. Do đó \(c=(.4,.4,.4)\). Tráo hai lần đọc ở ảnh thứ hai giữ nguyên Dot, các giá trị min/max từng ảnh và graph, nhưng tạo cạnh nối hai lần đọc ở mức .8. Lúc đó \(c=(.8,.8,.6)\). Phép dựng được kiểm tra bằng tìm thành phần trực tiếp, cùng với 16 cặp query–ảnh lấy bằng seed 42 trên mỗi split thật. Phép truyền chỉ đọc vector Dot và graph cố định sẽ có cùng đầu vào ở cả hai tình huống.

Phép hồi phục từ một mask và marker là cơ chế đã được dùng trong phân tích và phân đoạn ảnh. Phần đề xuất là tạo marker từ giao nhau hoặc trực tiếp chuyển lần đọc trên cạnh của gallery, và dùng nó để xác định xem hai cách đọc qua CLIP có một miền gallery chung ở từng ngưỡng hay không. Relative fuzzy connectedness dùng sức mạnh đường để phân vùng đối tượng từ các seed được cung cấp. Cơ chế này tạo seed trực tiếp từ hai lần đọc của một query và kiểm tra sự cùng có mặt trong thành phần. [Vincent, IEEE TIP 1993](https://telin.ugent.be/~sanja/StudentProj/Literature/vincent-93.pdf) · [Ciesielski et al., CVIU 2007](https://pmc.ncbi.nlm.nih.gov/articles/PMC2442428/)

Iscen truyền các điểm truy hồi trên manifold. ICFRR cập nhật thứ hạng cross-domain bằng thứ hạng gallery–gallery. Pareto-depth tìm thỏa hiệp giữa các truy vấn bằng quan hệ trội. FreeDom giải ngữ nghĩa ảnh truy vấn qua từ và STiTch đối sánh các tập thành phần giữa truy vấn và một ảnh. Các thiết lập đó cung cấp đối chiếu cho việc sử dụng graph, nhiều quan sát và biểu diễn trung gian. Đối tượng của phép thử hiện tại là thành phần gallery được cùng hai lần đọc của một truy vấn xác nhận. [Iscen et al., CVPR 2017](https://arxiv.org/abs/1611.05113) · [Hudson và Smith, ICFRR](https://arxiv.org/abs/2303.17703) · [Hsiao et al., Pareto-depth](https://web.eecs.umich.edu/~hero/Preprints/hsiao-tip2015_final.pdf) · [FreeDom, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/papers/Efthymiadis_Composed_Image_Retrieval_for_Training-Free_Domain_Conversion_WACV_2025_paper.pdf) · [STiTch, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Li_STiTch_Semantic_Transition_and_Transportation_in_Collaboration_for_Training-Free_Zero-Shot_CVPR_2026_paper.html)

## Kiểm chứng

- **Tráo danh tính lần đọc** ở một tập ảnh gallery chọn seed 42. Hai điểm, trung bình, giá trị nhỏ nhất, giá trị lớn nhất và graph của mỗi ảnh đều giữ nguyên. Phép tráo có thể tạo những cạnh đổi lần đọc giả. Lợi ích về tỷ lệ cùng lớp ở cùng số ảnh mới là bằng chứng thực địa cho vai trò của vị trí cạnh đổi lần đọc.
- **Marker từ cùng ảnh** đặt marker bằng \(d_i\) với cùng trần và cùng suy luận đến điểm cố định. Đối chứng này kiểm tra phần tăng thêm khi cho phép hai lần đọc có nhân chứng ở hai ảnh.
- **Dot và RGB diffusion** là các mốc truy hồi. Diffusion dùng chính graph 16 láng giềng hai chiều, cosine mũ ba chuẩn hóa từng hàng và restart 0,5 trong 16 vòng. Dot đủ 601 lấy lại từ tệp kết quả đã kiểm tra SHA và thứ tự query/gallery. Precision gốc được đối chiếu từng query. Các nhánh mới chạy hàm KD-SBIR gốc.

Các tham số được khóa trước khi chấm. Full unseen gồm 14.179 / 12.694 / 2.130 sketch và 17.100 / 12.552 / 27.262 ảnh cho Sketchy Ext-1, Ext-2 và TU-Berlin 30 lớp. Ext-1 và TU-Berlin dùng mAP@all/P@100. Ext-2 dùng mAP@200/P@200. Chạy trên Colab, seed 42, một CLIP ViT-B/32 đóng băng, một query độc lập và gallery không nhãn. Nhãn chỉ vào evaluator và phép đo sau khi có điểm. Kỳ vọng chính là tăng tỷ lệ đúng ở cùng số ảnh mới so với cả hai đối chứng cơ chế. Theo dõi thêm mAP, precision, số ảnh đi qua ngưỡng nhờ cạnh đổi lần đọc và số vòng GPU để đạt điểm cố định. TU-Berlin kiểm tra chuyển giữa dataset trong cùng chiều sketch → photo.

~~~sh
PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -u -m tf_sbir.paired_gallery_connectivity
~~~
