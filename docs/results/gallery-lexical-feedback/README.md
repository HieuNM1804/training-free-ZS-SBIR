# Một ảnh khác xác nhận từ đang giải thích ứng viên

Khi CLIP cho một sketch xác suất cao ở vài từ, diffusion RGB chỉ cho biết ảnh nào nhận nhiều bằng chứng sau khi gộp lân cận. Một câu hỏi khác là những ảnh ủng hộ **từng từ** có thực sự phù hợp với sketch hay không. Hướng này dùng chính từ đang đóng góp cho ảnh ứng viên để chọn tập ảnh cung cấp phản hồi. Mã hóa mỗi truy vấn độc lập với cùng CLIP ViT-B/32. Hai miền dùng hai bộ prompt đi theo cùng danh sách 20.931 từ Open Images.

## Từ đồ thị đến điểm truy hồi

1. Mỗi ảnh có phân bố từ $P_j$ từ CLIP. Tạo toán tử diffusion $T$ trên RGB, với 16 ảnh gần nhất, hệ số băng thông 0,36 và hai lượt cập nhật. Tính trước $B=TP$, phân bố từ sau diffusion của mỗi ảnh.
2. Với sketch có likelihood $l_w$, điểm $e_j=\sum_w B_{jw}l_w$ tái tạo baseline. Trong không gian từ, đây là một phép chấm tuyến tính. Không cần đọc sketch khác.
3. Với mỗi ứng viên $i$ và mỗi từ $w$, lấy trung bình các $e_j$ của những ảnh còn lại, có trọng số $B_{jw}$. Dùng giá trị này nhân với lượng bằng chứng $B_{iw}l_w$ ban đầu. Cộng tất cả các từ để xếp hạng ảnh. Một từ chung có nhiều ảnh hỗ trợ có mẫu số lớn tương ứng. Cấu hình giữ nguyên khi chuyển sang TU-Berlin.

$$
s_i(q)=\sum_w B_{iw}l_w\frac{\sum_{j\ne i} B_{jw}e_j}{\sum_{j\ne i} B_{jw}} .
$$

Đổi sketch có thể đổi cả từ đóng góp lẫn phản hồi của gallery cho từng từ đó. Điểm mới là bậc hai theo $l$; một bảng ảnh–từ tuyến tính cố định thường không thể thay nó cho mọi sketch. Giả thuyết ngược dùng hỗn hợp ảnh để tái tạo phân bố từ của sketch. Lượt đầu đúng bằng baseline; những lượt sau ưu tiên từ gallery đang giải thích thiếu.

## Kết quả

Mỗi ô là mAP / precision (%). Ext‑1 và TU dùng mAP@all / P@100. Ext‑2 dùng mAP@200 / P@200.

| Cấu hình | Sketchy Ext‑1 | Sketchy Ext‑2 | TU-Berlin, 30 lớp |
|---|---:|---:|---:|
| Khớp qua từ + diffusion RGB gốc | 81,86 / 81,33 | 74,19 / 72,68 | 79,16 / 80,24 |
| Xác nhận theo cùng từ, loại chính ứng viên | **82,56 / 81,46** | **74,07 / 72,73** | **79,77 / 80,62** |
| Cho phép chính ứng viên xác nhận | 82,55 / 81,46 | 74,05 / 72,73 | 79,77 / 80,62 |
| Không buộc hai tín hiệu cùng một từ | 82,33 / 81,39 | 74,07 / 72,76 | 79,10 / 80,71 |
| Chỉ dùng phản hồi từ gallery | 82,48 / 81,17 | 72,70 / 71,56 | 76,83 / 78,78 |
| Làm sắc likelihood của sketch | 79,56 / 78,18 | 71,75 / 70,29 | 78,76 / 78,89 |
| Xáo tên từ trong phản hồi gallery | 81,96 / 81,69 | 74,74 / 73,35 | 78,52 / 79,40 |
| Tái tạo ngược, hai lượt | 80,64 / 80,83 | 73,84 / 72,16 | 78,53 / 79,88 |

Cấu hình xác nhận theo từ đứng đầu ba ứng viên đã đặt trước theo trung bình mAP trên hai split Sketchy unseen. Mức tăng mAP so với baseline là +0,70 [0,37; 1,18], −0,13 [−1,06; 0,64] và +0,61 [0,15; 1,30] điểm. Các khoảng là bootstrap 95% theo lớp trên chênh lệch ghép cặp. Phần lớn mức tăng vẫn còn khi cho ứng viên tự xác nhận. Riêng TU, buộc xác nhận theo cùng từ tăng 0,67 [0,09; 1,38] điểm mAP so với nhân hai tín hiệu có thể đến từ các từ khác nhau. Chênh lệch với đối chứng này chưa tách khỏi 0 ở cả hai split Sketchy. Ext‑2 còn có trường hợp xáo tên từ cho mAP cao hơn cấu hình chọn. Tất cả phép so là thăm dò.

Điểm độ chính xác theo lớp và độ ổn định danh tính của ảnh đầu danh sách là những kết quả khác nhau. Trên TU, phương pháp đẩy 44 trường hợp từ sai sang đúng ở top‑1 và 61 từ đúng sang sai, kể cả khi mAP tăng. Tái tạo ngược sau bốn lượt giảm KL trung bình từ 1,27 xuống 0,94 (Ext‑1), từ 1,07 xuống 0,81 (Ext‑2) và từ 1,47 xuống 1,16 (TU); mAP giảm ở cả ba.

## Phạm vi của đóng góp

Phép kiểm tra phản thực tính chính xác thay đổi thứ hạng khi một tọa độ likelihood của truy vấn bị vô hiệu hóa. Trên mẫu cố định 304 truy vấn, 248 giữ lớp ảnh đầu dưới mọi phép bỏ một từ; chỉ 26 giữ cùng ảnh đầu. Có cả phép so với ICFRR trên cùng backbone và cùng điểm từ đầu vào. [Công thức và kiểm tra toàn gallery](../word-deletion-certificates/README.md) · [ICFRR](../icfrr-word-control/README.md)

FreeDom cung cấp tiền lệ dùng từ làm cầu nối xuyên miền. Wang và cộng sự cung cấp toán tử khuếch tán. Relevance models của Lavrenko và Croft dùng tài liệu trong bộ sưu tập để ước lượng phân bố từ liên quan tới truy vấn. Ở đây ảnh gallery đảm nhận vai trò của tài liệu. Đề xuất ở đây là điều kiện hóa nguồn phản hồi lên cùng từ đang giải thích ảnh ứng viên và kiểm tra chính xác ảnh hưởng của từ đó lên toàn bộ thứ hạng thông qua hai lượt sử dụng sketch. Đây là ba nguồn cơ chế trực tiếp. [FreeDom, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Efthymiadis_Composed_Image_Retrieval_for_Training-Free_Domain_Conversion_WACV_2025_paper.html) · [Wang et al., CVPR 2012](https://pages.ucsd.edu/~ztu/publication/cvpr12_CrDP.pdf) · [Lavrenko và Croft, SIGIR 2001](https://www.inf.ed.ac.uk/publications/report/1190.html)

Phản hồi bằng đồng xuất hiện khái niệm và chứng nhận độ ổn định của thứ hạng đều đã có trong các bài toán gần. Rahman và cộng sự dùng mở rộng truy vấn ảnh từ các khái niệm cục bộ/toàn cục. ICFRR khai thác gallery cùng miền để xếp lại truy hồi xuyên miền. Kakizaki và cộng sự chứng nhận độ bền của truy hồi ảnh đối với nhiễu; CertDR chứng nhận top‑K cho mô hình xếp hạng tài liệu dưới thay từ. Phần cần đánh giá về novelty là cơ chế cụ thể và phép chứng nhận khi vô hiệu hóa một khái niệm ẩn ở phía sketch. [Rahman et al.](https://pubmed.ncbi.nlm.nih.gov/21822350/) · [ICFRR](https://arxiv.org/html/2303.17703v2) · [Kakizaki et al., WACV 2023](https://openaccess.thecvf.com/content/WACV2023/html/Kakizaki_Certified_Defense_for_Content_Based_Image_Retrieval_WACV_2023_paper.html) · [CertDR, CIKM 2022](https://doi.org/10.1145/3511808.3557256)

Đã chấm mọi truy vấn bằng hàm KD-SBIR gốc trên Colab L4. TU có các kết quả từ những thí nghiệm trước. Đợt này khóa ứng viên theo Sketchy trước TU. Hai benchmark cùng dùng sketch → photo. Phép kiểm chứng hiện có chỉ thiết lập tính chuyển giao giữa bộ dữ liệu trong cặp miền đó. [Mã](../../../tf_sbir/gallery_lexical_feedback.py) · [Thiết kế](../../GALLERY_LEXICAL_FEEDBACK_PROTOCOL.md) · [Khóa](freeze.json) · [Ext‑1](sketchy_1.json) · [Ext‑2](sketchy_2.json) · [TU](tuberlin.json) · [Dữ liệu từng truy vấn](per-query.zip)
