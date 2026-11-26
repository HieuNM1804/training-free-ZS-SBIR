# Kiểm chứng một phép đổi tên bằng can thiệp vào prompt miền

## Từ lỗi thật đến giả thuyết

Hai tên có thể phản ứng giống nhau khi che vùng dù không cùng chỉ một vật thể. Trên Sketchy Ext-1, một sketch chiếc cốc đã đưa ảnh cây lên đầu khi “mug” và “palm tree” cùng trả lời một bộ câu hỏi che vùng. Việc chỉ quan sát tương quan trong ảnh chưa xác định **vì sao** tên khác nhau giữa hai miền. Câu hỏi của phương pháp này là liệu chính sự thay đổi prompt miền, khi giữ nguyên ảnh, có thể giải thích một phép thay tên cụ thể theo cả hai chiều hay không.

Giữ nguyên OpenAI CLIP ViT-B/32 đóng băng và 601 từ công khai. Hai cách hỏi duy nhất là “a drawing of a {word}.” và “a photo of the {word}.”. Khác với Dot chỉ dùng cách hỏi bản địa của mỗi ảnh, thí nghiệm đọc mỗi ảnh qua cả hai bộ text embedding đã tính trước. Không cần chạy lại encoder ảnh, tìm vùng, dùng sketch khác hay lấy thống kê từ gallery. Can thiệp vào prompt là phép can thiệp vào bộ đọc của CLIP. Nó không tái tạo quá trình con người vẽ ảnh.

## Suy luận một cặp

Cho $a_I,b_I$ là phân bố 601 tên của ảnh $I$ dưới prompt nguồn và đích. Dùng cùng một vector nhiễu Gumbel chuẩn $g$ ở hai lần đọc của chính ảnh:
$$
U_I=\arg\max_w(\log a_I(w)+g_w),\qquad V_I=\arg\max_w(\log b_I(w)+g_w).
$$
Gọi $T_I(u,v)=P(U_I=u,V_I=v)$. Với hai ảnh, các vector nhiễu độc lập. $g$ mô tả một lựa chọn từ giả định. Hai posterior quan sát được không tự xác định được phân bố chuyển tên; việc dùng chung nhiễu là một giả thuyết cấu trúc có thể bị bác bỏ bởi đối chứng. [Oberst & Sontag, ICML 2019](https://proceedings.mlr.press/v97/oberst19a.html) · [Lorberbom et al., NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/e143c01e314f7b950daca31188cb5d0f-Paper.pdf)

Rút tên bản địa $U_X=u$ của sketch và $V_Y=v$ của ảnh. Hai tên khớp nếu $u=v$. Khi $u\ne v$, truy vấn cần chọn $v$ nếu đổi sang prompt ảnh, đồng thời ảnh cần chọn $u$ nếu đổi về prompt sketch, với nhiễu của mỗi ảnh giữ cố định. Đó là một biến cố có hai trường hợp loại trừ nhau. Xác suất của nó là
$$
K(X,Y)=a_X^\top b_Y+\sum_{u\ne v}T_X(u,v)T_Y(u,v).
$$
Đối với một cặp tên khác nhau, xác suất chấp nhận sau khi đã quan sát hai tên bản địa là $T_X(u,v)T_Y(u,v)/(a_X(u)b_Y(v))$. Vì vậy hệ số giữa phần trùng tên và phần thay tên được xác định bởi biến cố, không cần chọn bằng nhãn unseen. Nếu hai prompt cho cùng posterior ở mọi ảnh, $K$ trở về Dot. Một phép đổi tên chỉ có thể xuất hiện khi tỷ lệ $b_v/a_v$ cao hơn $b_u/a_u$ ở **cả hai ảnh**.

## Tính chính xác và lưu trữ

Đặt $r_i=a_i/b_i$, $C(z)=\sum_i\max(a_i,b_i z)$ và $H(z)=\int_{\min r}^z C(t)^{-2}dt$. Với từng ảnh:
$$
d_i=T(i,i)=a_i/C(r_i),\qquad
T(u,v)=a_u b_v[H(r_u)-H(r_v)]_+ \quad(u\ne v).
$$
Chứng minh dùng các đồng hồ $E_i$ độc lập phân bố mũ tốc độ 1. Người thắng là đồng hồ nhỏ nhất sau khi chia cho $a_i$ hoặc $b_i$. Với hai người thắng khác nhau $u,v$, viết hai thời gian thắng là $t,zt$. Điều kiện cạnh tranh yêu cầu $r_v<z<r_u$. Mật độ sau khi lấy tích phân theo $t$ là $a_u b_v/C(z)^2$. Khi cùng người thắng, tích phân một chiều cho $a_i/C(r_i)$. Giữa hai tỷ lệ đã sắp, $C$ tuyến tính nên phần tăng của $H$ bằng $(r_{k+1}-r_k)/(C(r_k)C(r_{k+1}))$. Toàn bộ ma trận 601×601 vì thế được mã hóa bằng số thực tuyến tính theo kích thước từ vựng. Xác suất chuyển trạng thái dạng đóng trong mô hình logit đã có trong kinh tế lượng. [Nghiên cứu về xác suất chuyển lựa chọn, mục 4.1, công thức 19](https://thema.u-cergy.fr/IMG/pdf/2021-11.pdf) · [Analytical Prediction of Transitions Probabilities in the Conditional Logit Model](https://docs.iza.org/dp1015.pdf)

**Phần thuật toán truy hồi.** Đặt $h_I(w)=H_I(r_w)$, $A_u=a_X(u)a_Y(u)$, $B_v=b_X(v)b_Y(v)$. Khối lượng thay tên là tổng $A_u B_v(h_X(u)-h_X(v))(h_Y(u)-h_Y(v))$ trên các $v$ đứng trước $u$ ở cả hai thứ tự $h$. Quét theo thứ tự của $X$, dùng cây Fenwick theo thứ tự của $Y$ để duy trì bốn tổng tiền tố của $B_v$, $B_v h_X(v)$, $B_v h_Y(v)$, $B_v h_X(v)h_Y(v)$. Mỗi tên cần một truy vấn và một cập nhật. Do đó có thể tính chính xác trong $O(V\log V)$ thời gian và $O(V)$ bộ nhớ tạm cho một cặp, với $V=601$. Cách chấm hàng loạt trên GPU trong thí nghiệm sử dụng các khối tích ma trận trên các cặp tên và kiểm tra bằng lời giải tiền tố độc lập.

Với $D=a_X^\top b_Y$, $r'_I=a_I-d_I$, $c'_I=b_I-d_I$, ta có cận
$$
D\le K\le D+\min\{(r'_X)^\top r'_Y,(c'_X)^\top c'_Y\}.
$$
Mỗi cận trên thu được bằng cách bỏ một trong hai điều kiện đồng nhất đầu mút của phép đổi tên. Chúng cho phép lọc ứng viên bằng các vector 601 chiều rồi mới tính các phép thay tên có thể ảnh hưởng top-K. Số ảnh phải chấm và thời gian thực cần được kiểm tra trên đúng gallery.

## Đối chứng có thể bác bỏ cơ chế

- Dùng nhiễu độc lập khi đổi prompt nhưng giữ đúng hai phân bố theo miền. Trường hợp khớp tên bản địa dùng cùng Dot.
- Giữ **chính xác cả hai phân bố biên, mọi xác suất giữ nguyên từng tên và mọi hướng thay tên bị cấm**. Trong phần đã đổi tên, ghép các khoảng phân vị của nguồn dư và đích dư theo thứ tự tỷ lệ $a/b$. Cặp được gán khối lượng bằng giao hai khoảng. Do thứ tự ưu thế của các biên, khối lượng chỉ đi theo những hướng được phép. Kiểm tra số học riêng trên cả hai miền. Đây là đối chứng mạnh cho việc Gumbel phân bổ xác suất giữa những cách đổi tên *đều* có thể xảy ra.
- Chỉ yêu cầu xác nhận từ một phía để kiểm tra phần đóng góp của yêu cầu cả hai ảnh giải thích được cùng phép đổi tên. Kiểm tra thêm xác suất cùng cả hai tên kể cả khi tên bản địa đã trùng.
- Kiểm tra trên ảnh nguồn có ID trong Sketchy, khi sketch và ảnh nguồn có tên bản địa đứng đầu khác nhau. So với một ảnh khác có **đúng cùng tên đích đứng đầu** và có Dot gần nhất, trong lớp hoặc khác lớp. Nhãn và ID nguồn chỉ dùng sau khi đã có điểm. Chấm retrieval trên toàn bộ Sketchy Ext-1, Sketchy Ext-2 và TU-Berlin 30 lớp với evaluator KD-SBIR, cùng seed 42, không điều chỉnh siêu tham số. Chênh lệch độ đúng lớp ở cùng số ảnh mới vào top-K là đối chứng chính.
- Một lời giải ngoài đường chéo có công thức gọn hoặc một vài ví dụ đúng không đủ chứng minh cơ chế. Nếu đối chứng giữ cả biên lẫn hướng đạt cùng kết quả, phần có thể bảo vệ sẽ hẹp hơn phát biểu rằng coupling Gumbel giúp chọn ảnh đúng.

## Khoảng cách đến prior art

QBSE so sánh các phân bố tên trên concept có sẵn. FreeDom đi qua các tên do ảnh proxy ủng hộ để truy hồi theo miền đích. SpLIP học prompt chia sẻ cho ZS-SBIR, CLAIR ước lượng phép căn chỉnh hình học từ text, BASIC phối hợp bằng chứng text và ảnh, STiTch dùng chuyển tiếp và transport giữa caption và ảnh. Ở đây đối tượng cần suy ra là một cặp lựa chọn có điều kiện **trên cùng mỗi ảnh khi chỉ đổi prompt**. Đóng góp ứng viên là biến cố khớp phản thực hai chiều cho hai ảnh, cách tính chính xác bằng các thứ tự ưu thế và khả năng kiểm tra từng phép đổi tên dưới một placebo giữ các biên. Mô hình Gumbel, công thức logit và cấu trúc dữ liệu tổng tiền tố đều có tiền lệ. Các nguồn cơ chế trực tiếp là công trình Gumbel-max, công thức chuyển lựa chọn logit và baseline truy hồi bằng posterior. [QBSE](https://doi.org/10.1109/TMM.2007.900138) · [FreeDom](https://openaccess.thecvf.com/content/WACV2025/papers/Efthymiadis_Composed_Image_Retrieval_for_Training-Free_Domain_Conversion_WACV_2025_paper.pdf) · [SpLIP](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/3426_ECCV_2024_paper.php) · [CLAIR](https://bmvc2025.bmva.org/proceedings/126/) · [BASIC](https://papers.neurips.cc/paper_files/paper/2025/hash/6391a02f02dc529cfa3f242fb4343c6a-Abstract-Conference.html) · [STiTch](https://openaccess.thecvf.com/content/CVPR2026/html/Li_STiTch_Semantic_Transition_and_Transportation_in_Collaboration_for_Training-Free_Zero-Shot_CVPR_2026_paper.html)
