# Ước lượng khái niệm của ảnh qua tần suất từ và ảnh lân cận

Mỗi ảnh được mô tả bằng phân bố trên **20.931 từ công khai của Open Images**. Với từ $w$ và ảnh gallery $g_j$, đặt $a_j(w)\propto\exp(\tau\,g_j^\top t_{\text{photo},w})$, trong đó $\tau$ là logit scale của CLIP ViT-B/32. Hai prompt ảnh và hai prompt sketch dùng chung danh sách từ. FreeDom cho tiền lệ nhận diện nội dung xuyên miền bằng từ vựng rời rạc. [FreeDom, WACV 2025](https://arxiv.org/abs/2412.03297)

Từ gallery ảnh không nhãn, ước lượng phân bố tần suất $\pi$ bằng cách cực đại hóa $\sum_j \log(\sum_w \pi_w a_j(w))$. Mọi embedding từ và tham số CLIP được đóng băng. Sau đó xác suất từ cho từng ảnh là

$$
b_j(w)=\frac{\pi_w a_j(w)}{\sum_v \pi_v a_j(v)}.
$$

Việc ước lượng prior đích bằng maximum likelihood có tiền lệ trong nghiên cứu label shift. Ở đây các thành phần là khái niệm công khai, rộng hơn danh mục lớp của benchmark. [Alexandari và cộng sự, ICML 2020](https://proceedings.mlr.press/v119/alexandari20a.html)

Với ảnh $i$, lấy các ảnh gần nhất trong không gian RGB của CLIP, loại chính ảnh đó. Phân bố dự báo là $r_i(w)=\sum_{j\in N_i}\alpha_{ij} b_j(w)$. Một sketch $q$ xếp ảnh $i$ bằng điểm $s(q,i)=\sum_w L_q(w)r_i(w)$, trong đó $L_q(w)=\exp(\tau\,q^\top t_{\text{sketch},w})$. Có thể chuẩn hóa mỗi hàng $L_q$ bằng bất kỳ thừa số dương nào của riêng sketch. Việc dùng ngữ cảnh để ước lượng phân bố của một tài liệu lấy từ cluster-based language models trong truy hồi văn bản. Trong cách chuyển sang SBIR này, mỗi ảnh có một lân cận riêng, không cần phân hoạch gallery thành các lớp. [Liu và Croft, SIGIR 2004](https://ciir.cs.umass.edu/pubfiles/ir-347.pdf)

Cấu hình chính chọn **256 láng giềng**. Các trọng số $\alpha_{ij}$ là Gaussian chuẩn hóa trong lân cận, dùng khoảng cách $d_{ij}^2=2(1-\cos(g_i,g_j))$ với mẫu số $0.36\,\bar d^2$. Bề rộng được tính từ khoảng cách láng giềng của gallery hiện tại. Hệ số số học 0,36 kế thừa từ đối chứng diffusion trước đó. Công thức chính chỉ lấy trung bình lân cận một lần. Việc đổi sang trung bình đều là đối chứng cho lựa chọn trọng số này. [Nguồn của quy ước Gaussian, Wang và cộng sự, CVPR 2012](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/cvpr12.pdf)

| Phép can thiệp | Sketchy Ext-1 mAP@all | Sketchy Ext-2 mAP@200 | TU-Berlin mAP@all |
|---|---:|---:|---:|
| Hiệu chỉnh posterior rồi gom ảnh lân cận | 82,65 | 74,79 | 78,32 |
| Bỏ lân cận, chỉ giữ posterior của từng ảnh | 80,60 | 74,02 | 76,59 |
| Giữ lân cận với prior từ đều | 81,54 | 73,74 | 78,95 |
| Gom ảnh lân cận trước rồi hiệu chỉnh | 81,93 | 74,94 | 75,99 |
| Dùng 256 láng giềng với trọng số đều | 82,73 | 74,39 | 77,82 |
| Xáo danh tính ảnh lân cận | 5,09 | 8,58 | 4,23 |

Hai split unseen của Sketchy dùng để chọn cấu hình thăm dò. Trên TU, prior ước lượng lại từ gallery không nhãn của TU, cùng toàn bộ tham số đã khóa. [Mã của phương pháp tại bản chạy](https://github.com/HieuNM1804/training-free-ZS-SBIR/blob/a8a73530a993133cb4f8058bfe61e7f70cb0f68f/tf_sbir/neighborhood_calibrated_likelihood.py) · [Báo cáo các phương pháp và phép kiểm tra](https://github.com/HieuNM1804/training-free-ZS-SBIR/blob/research/map-retrieval-analysis/docs/MAP_FOLLOWUP_RESULTS_VI.md)
