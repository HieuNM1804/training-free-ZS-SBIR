# Các vùng của cùng một sketch đang cung cấp bao nhiêu bằng chứng độc lập?

Một sketch có thể cho cùng một tên CLIP khi nhìn toàn hình và khi nhìn một phần hình. Tích năm điểm có thể đếm lặp cùng một bằng chứng. Ép cả năm vùng cùng chỉ một khái niệm lại có thể làm mất các chi tiết bổ sung. Giả thuyết được kiểm tra là cho phép một số vùng cùng dựa trên một khái niệm, đồng thời cho các vùng khác có bằng chứng riêng, sẽ cải thiện thứ hạng của ảnh đúng.

## Mô hình

Mỗi ảnh gallery $y$ cho một phân bố $\theta_y$ trên 601 từ qua CLIP ViT-B/32. Mỗi truy vấn có năm quan sát gồm toàn sketch và bốn cửa sổ ở bốn góc. Cùng một cửa sổ được đọc theo hướng gốc và hướng lật ngang, sau đó lấy trung bình hai phân bố từ. Gọi $\ell_j(w)$ là phản hồi mềm của quan sát $j$ với từ $w$. Việc dùng $\ell_j$ như một likelihood là giả định của mô hình.

Để chấm một ảnh, lấy một phân bố tiềm ẩn $z$ từ $\mathrm{Dirichlet}(\alpha\theta_y)$. Mỗi vùng rút từ theo $z$. Likelihood sau khi tích phân $z$ là

$$
L(q,y)=\mathbb E_z\prod_{j=1}^{5}\langle\ell_j,z\rangle
=\frac{1}{(\alpha)^{\overline{5}}}\sum_{\pi}
\alpha^{|\pi|}\prod_{B\in\pi}
\left[(|B|-1)!\sum_w\theta_y(w)\prod_{j\in B}\ell_j(w)\right].
$$

Ở đây $\pi$ là một phân hoạch của năm quan sát, mỗi khối $B$ dùng chung một lần rút tên, và $(\alpha)^{\overline{5}}=\alpha(\alpha+1)\cdots(\alpha+4)$. Mã tính đủ 52 phân hoạch. Với năm quan sát có 31 tập con khác rỗng. Tích theo từ cho từng tập con được tính một lần cho mỗi sketch và dùng lại với toàn gallery. Một truy vấn dùng 10 lượt mã hóa ảnh gồm cả các bản lật. Baseline dùng 2 lượt. [Mã phép tính](../tf_sbir/exchangeable_evidence.py)

## Phép thử có thể bác bỏ giả thuyết

- **Các vùng rút độc lập** cho $\prod_j\langle\ell_j,\theta_y\rangle$. **Cả năm dùng một lần rút** cho $\sum_w\theta_y(w)\prod_j\ell_j(w)$. Hai đối chứng dùng chính các crop và cùng CLIP.
- **Hỗn hợp của hai trường hợp cực hạn** cho cả năm vùng dùng chung với xác suất $1/(1+\alpha)$, còn lại cho chúng rút độc lập. Mỗi cặp vùng có cùng xác suất chia sẻ lần rút ban đầu như trong mô hình chính. Khác biệt khi xét từ ba vùng trở lên giúp kiểm tra giá trị của việc chia sẻ từng phần.
- **Lấy trung bình các crop** cho trung bình của năm điểm Dot. Đối chứng này đo phần thay đổi do nhìn ảnh ở nhiều vùng. **Dot toàn sketch** là mốc cho yêu cầu tăng ít nhất bốn điểm mAP tuyệt đối.
- Chọn cạnh cửa sổ trong $\{0.5,0.75\}$ lần chiều tương ứng của hình và $\alpha$ trong $\{0.125,0.5,2,8,32\}$. Dùng trung bình mAP hai split Sketchy unseen, phân xử hòa bằng precision, rồi khóa trước TU-Berlin. Seed là 42. Các phép đo chỉ kiểm chứng chiều sketch → photo. [Mã giao thức và đối chứng](../tf_sbir/exchangeable_study.py)

## Quan hệ với prior art

Madsen và cộng sự dùng phân bố Dirichlet để mô hình hóa sự lặp từ trong văn bản. Cummins và cộng sự áp dụng mô hình Pólya cho xếp hạng tài liệu. Một tiền lệ thị giác cũng dùng sự lặp của đặc trưng cục bộ trong mô hình chủ đề đa phương tiện. Phần đáng kiểm chứng ở đây là cho phép **tên chưa biết của các vùng ở miền nguồn** chia sẻ bằng chứng khi kiểm tra **phân bố từ của từng ảnh đích**, cùng đối chứng khớp xác suất chia sẻ ở cấp cặp. Công thức moment Dirichlet và hiện tượng lặp đều đã có tiền lệ. [Madsen et al., ICML 2005](https://icml.cc/Conferences/2005/proceedings/papers/069_WordBursting_MadsenEtAl.pdf) · [Cummins et al., Pólya Urn Document Language Model](https://arxiv.org/abs/1502.00804) · [Multimedia Topic Models Considering Burstiness of Local Features](https://doi.org/10.1587/transinf.E97.D.714)
