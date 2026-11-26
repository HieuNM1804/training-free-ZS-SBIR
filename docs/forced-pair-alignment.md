# Chi phí căn chỉnh khi buộc một cặp sketch–ảnh khớp nhau

## Giả thuyết

Trong baseline, hai ảnh được so bằng Dot trên 601 xác suất từ. Dot xem danh tính từng mục từ là tương ứng cố định giữa hai miền. Căn chỉnh trực giao toàn cục bằng các cặp prompt còn cho phép so những mô tả gần nghĩa, nhưng dùng một nghiệm tối ưu duy nhất. Khi bộ từ ràng buộc các hướng mạnh yếu khác nhau, khoảng cách sau nghiệm đó bỏ qua mức độ tự do còn lại. Câu hỏi là một cặp cùng lớp có dễ ép khớp hơn một cặp khác lớp, tính theo phần tổn thất phát sinh trên *toàn bộ* bộ từ neo, sau khi đã kiểm soát khoảng cách ngữ nghĩa ban đầu hay không.

## Một bài toán suy luận cho từng cặp

Với cùng 601 từ, mã hóa mỗi từ qua hai prompt đã khóa. Lấy các vector đơn vị, trừ trung bình theo từng miền để thu được $a_w,b_w\in\mathbb R^{512}$. Với posterior ngữ nghĩa $q,p$ của một sketch và một ảnh, lập hai vector đơn vị $x=\operatorname{norm}(\sum_w q_w a_w)$, $y=\operatorname{norm}(\sum_w p_w b_w)$. Đây là hướng dịch chuyển của trung bình embedding từ so với bộ từ đều. Một posterior hoàn toàn đều không xác định một hướng. Kiểm tra độ dài trước chuẩn hóa trên dữ liệu là điều kiện để diễn giải chi phí này. Baseline và mọi đối chứng chia sẻ posterior gốc. Các đối chứng hình học còn chia sẻ $x,y$.

Đặt $M=601^{-1}\sum_w a_w b_w^\top$. Mỗi ứng viên đặt ra một giả thuyết riêng $R y=x$, với $R\in O(512)$. Chi phí của giả thuyết là phần tổn thất thêm trên bộ từ khi đòi cặp đó khớp chính xác,

$$
\Delta(x,y)=\min_{R:Ry=x}L(R)-\min_R L(R),\qquad
L(R)=\frac1{2\cdot601}\sum_w\|a_w-Rb_w\|^2.
$$

Xếp tăng dần theo $\Delta$. Thuật toán có một mục tiêu duy nhất, không cần gallery có láng giềng cùng lớp. Mỗi ảnh thêm vào gallery nhận một chi phí độc lập. $R$ được phép phản xạ. Phép biến đổi căn đúng hai vector hướng là một giả định mô hình mạnh cho truy hồi theo lớp, nơi hai ảnh đúng có thể có nội dung khác nhau.

## Nghiệm và chi phí tính

Viết $P_z=I-zz^\top$. Một phép biến đổi khả thi ánh xạ $y$ đến $x$ và còn một ánh xạ trực giao tự do từ $y^\perp$ đến $x^\perp$. Tối ưu phần sau bằng Procrustes thu được

$$
\Delta=\|M\|_* - x^\top M y - \|P_x M P_y\|_*.
$$

Nếu đổi cơ sở trực giao độc lập ở hai miền, mọi hạng giữ nguyên. Nếu bỏ ràng buộc của ứng viên thì $\Delta=0$ với mọi cặp. Các hướng có trị kỳ dị nhỏ cho phép thay đổi ánh xạ ít làm suy giảm độ khớp của từ neo. Bất biến với đổi cơ sở là một đảm bảo đại số. Việc đúng lớp có đi kèm với chi phí nhỏ vẫn là giả thuyết phải đo.

Không cần giải SVD kích thước 512 cho từng cặp. Tính $M=U\Sigma V^\top$ một lần rồi chuyển $x,y$ sang hai cơ sở đó. Cho $t>0$, đặt $A_x(t)=\sum_k x_k^2 t^2/(t^2+\sigma_k^2)$, $A_y$ tương tự và $B_{xy}(t)=\sum_k x_k y_k t\sigma_k/(t^2+\sigma_k^2)$. Khi đó

$$
-\Delta = x^\top\Sigma y + \frac{1}{\pi}\int_0^\infty
\log\{A_x(t)A_y(t)+B_{xy}(t)^2\}\,dt .
$$

Chứng minh dùng đẳng thức $\int_0^\infty \log(1+s^2/t^2)\,dt=\pi s$ cho $s\ge 0$. Xét ma trận khối có hai khối chéo $I$, hai khối ngoài đường chéo $M/t$ và $-M^\top/t$. Định thức của nó là $\prod_k(1+\sigma_k^2/t^2)$. Trong cơ sở bắt đầu bằng $x$ và $y$, xóa hai tọa độ đầu của hai miền để thu định thức của phần nén. Theo đẳng thức phần bù của Jacobi, tỷ số định thức bằng định thức khối $2\times2$ của nghịch đảo ở hai tọa độ đã xóa, tức $A_xA_y+B_{xy}^2$. Tích phân của log tỷ số cho chênh lệch hai nuclear norm. Các đẳng thức là kết quả toán học đã biết và được vận dụng ở đây.

Mã hóa ma trận và mọi score theo đơn vị $\sigma_{\max}=1$. Rời rạc hóa tích phân bằng 32 nút Gauss–Legendre trên biến $z\in(0,1)$ với $t=\sigma_{\rm RMS} z/(1-z)$. Chọn 32 nút trước khi chấm. Có $O(L d)$ phép tính cho một cặp với $L$ nút, và mọi cặp ở một nút được tính theo batch bằng nhân ma trận. Số phép tính online cao hơn Dot 601 chiều. Đối chiếu với SVD trực tiếp và so 32/64 nút cần được báo riêng.

## Phép thử có thể bác bỏ cơ chế

1. Đối chứng căn chỉnh một lần dùng cosine $x^\top y$ trong hai cơ sở SVD. Thêm đối chứng song tuyến tính $x^\top\Sigma y$ để đo phần tiếp cận chỉ thông qua trị kỳ dị.
2. Cô lập phần phụ thuộc đồng thời vào hai ảnh bằng cách bỏ $B_{xy}^2$ trong tích phân. Hai phần hiệu chỉnh của từng ảnh và toàn bộ hạng song tuyến tính vẫn giữ nguyên. Nhánh này cho phép so ảnh mới ở cùng số lượng và xác định đúng/sai trên những cặp ảnh trong top của Dot, có điểm đối chứng gần nhau nhưng bị thuật toán đảo thứ hạng.
3. Tráo trị kỳ dị với seed 42, giữ nguyên hai cơ sở, phổ và ánh xạ tối ưu toàn cục. Phép thử đánh giá việc **gán đúng** độ cứng của neo cho từng hướng.
4. Đối chiếu khả năng tối ưu phần bù với hai ánh xạ khả thi tính trực tiếp. Trong hai cơ sở SVD, ký hiệu $c=x^\top y$, $b=x^\top\Sigma y$, $n=x^\top\Sigma x+y^\top\Sigma y$. Phép quay trong mặt phẳng hai vector có giá $(n-2cb)/(1+c)$. Phản xạ Householder có giá $(n-2b)/(1-c)$. Cả hai đều đưa $y$ tới $x$ ở các điểm không suy biến. Nghiệm tối ưu phải có giá nhỏ hơn hoặc bằng giá của từng ánh xạ. Mức chênh về thứ hạng cho biết có cần tối ưu thêm các chiều còn lại hay không.

Giữ nguyên encoder OpenAI CLIP ViT-B/32, các từ và hai prompt, hai lần đọc cùng một sketch, seed 42 và hàm chấm KD-SBIR. Chỉ nhãn benchmark sau khi xếp hạng mới tham gia đánh giá. Hai split Sketchy và TU-Berlin 30 lớp đều chạy full unseen với một cấu hình. Protocol đo khả năng chuyển giữa hai bộ dữ liệu cùng chiều sketch–photo. Ghi rõ cả chất lượng của ảnh mới, mAP, precision, chi phí và vấn đề nếu giả thuyết bị bác bỏ.

