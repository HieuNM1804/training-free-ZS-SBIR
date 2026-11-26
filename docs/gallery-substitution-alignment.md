# Thay truy vấn vào một cặp quan sát của gallery

## Câu hỏi và giả định

Khi kiểm tra một cặp giả định, cần mốc đối chiếu nào để diễn giải mức sai lệch? Trong thí nghiệm căn chỉnh cưỡng bức bằng 601 cặp từ, một ảnh có thể đạt sai số thấp vì bộ neo dễ biến đổi ở hướng của nó. Phép thử ở đây xây dựng một đối chứng riêng từ **hai lần đọc chính ảnh gallery đó**, cùng encoder ảnh, qua hai bộ prompt đã khóa. Giả thuyết là thay một lần đọc bằng sketch đúng lớp sẽ phá vỡ độ khớp của những cặp còn lại ít hơn khi thay bằng sketch khác lớp. Lần đọc RGB với prompt drawing là một phép đo dưới bộ từ của miền nguồn. Độ giống nó với một sketch thực là giả định cần kiểm chứng. Phép thay là giả định trong bài toán tối ưu. Nó không thiết lập quan hệ nhân quả từ ảnh tới sketch.

Dùng cả 601 thành phần của posterior. Với một vector xác suất $p$, tọa độ $\sqrt p$ có độ dài 1, phép đổi tọa độ có nghịch đảo $v\mapsto v^2$ và khoảng cách Euclid giữa hai vector tương ứng khoảng cách Hellinger tới một hằng số. Chuyển tọa độ bảo toàn phân bố. Khoảng cách và thứ hạng Dot vẫn có thể thay đổi. Đối chứng Hellinger trên cùng đầu vào đo riêng ảnh hưởng của hệ tọa độ.

## Một phép thay, cùng một bài toán tối ưu

Gọi $a_j,b_j$ là hai lần đọc ảnh $j$ trong các bộ prompt nguồn và đích, $x$ là lần đọc sketch. Các vector đều là căn bậc hai của posterior 601 từ. Định nghĩa $M=N^{-1}\sum_j a_jb_j^\top$, $M_{-i}=M-a_i b_i^\top/N$, và
$$
V(K,z,b)=\max_{R^\top R=I,\;Rb=z}\langle R,K\rangle_F .
$$
Mức thay đổi của độ khớp trên những ảnh còn lại là
$$
S_i(x)=V(M_{-i},x,b_i)-V(M_{-i},a_i,b_i).
$$

Xếp hạng theo $S_i$ giảm dần. Hai giả thuyết dùng cùng gallery còn lại và cùng số bậc tự do. Giả thuyết đầu buộc lần đọc đích của ứng viên khớp với sketch. Giả thuyết sau buộc nó khớp với lần đọc nguồn vốn quan sát được của chính nó. Điểm là một chênh lệch có dấu. Một ảnh có lần đọc nguồn và đích bất nhất có thể đạt điểm cao khi sketch thay thế lần đọc kém phù hợp ban đầu. Điều kiện $S_i(a_i)=0$ không đảm bảo ảnh $i$ đứng đầu.

Với $P_z=I-zz^\top$, viết
$$
V(K,z,b)=z^\top Kb+\|P_z K P_b\|_* .
$$
Khi ép $Rb_i=z$, phần đóng góp của cặp bị giữ lại là $a_i^\top z/N$ ở **mọi** nghiệm khả thi. Do đó chỉ cần một SVD cho toàn gallery,
$$
S_i(x)=V(M,x,b_i)-V(M,a_i,b_i)+(1-a_i^\top x)/N .
$$
Hai lần tối ưu không có hệ số pha trộn. Hệ số $1/N$ là hệ quả của định nghĩa ma trận trung bình. Trong tọa độ Hellinger không âm, hiệu chỉnh cuối nằm trong $[0,1/N]$. Vì vậy loại đúng một ảnh khỏi gallery chỉ có thể thay đổi chênh giữa hai ứng viên tối đa $1/N$ với đơn vị chưa chuẩn hóa. Phần lớn một mức tăng lớn so với chi phí chưa đối chiếu có thể đến từ số hạng chỉ phụ thuộc vào ảnh. Việc gán đúng tên cơ chế cho mỗi mức tăng cần đối chứng trực tiếp.

## Thử nghiệm và điều kiện bác bỏ

Sử dụng cố định ViT-B/32, 601 từ, seed 42, hai prompt mặc định, evaluator KD-SBIR và ba split full unseen. Không có hệ số được chọn bằng kết quả trên split. Mỗi sketch được xử lý riêng. Việc ước lượng hình học dùng duy nhất gallery RGB không nhãn của split hiện tại. Cả hai benchmark cùng đo sketch sang ảnh.

- **Hellinger và ánh xạ cố định** cho biết bản thân việc đổi tọa độ và một ánh xạ khớp cùng gallery mạnh tới đâu.
- **Chi phí chưa so với lần đọc thật** giữ nguyên phép tối ưu đầy đủ và cùng ma trận gallery. So với nó đo giá trị gia tăng khi có mốc của từng ảnh.
- **Phép quay hoặc phản xạ khả thi** bắt cả hai giả thuyết thỏa điều kiện khớp, nhưng chỉ dùng một trong hai ánh xạ có công thức tường minh. Phép so kiểm tra giá trị của tối ưu mọi hướng còn lại.
- **Hoán vị mốc tự nhiên** giữ nguyên toàn bộ từ, query, ma trận và các chi phí; tráo danh tính chi phí đối chứng giữa các ảnh có mức đồng thuận và mức tập trung posterior gần nhau. Với một seed, đây là đối chứng có điều kiện, không phải kiểm định hoán vị nhiều lần.
- Xét cùng số ảnh mới trong top-K của từng sketch. So nhãn sau khi xếp hạng. Ảnh thay thế top-K của Dot được so đồng thời với chính các ảnh bị đẩy ra. Nếu thành phần riêng không thắng các đối chứng trên TU-Berlin với cấu hình đã khóa, chưa đủ căn cứ để chọn thành phần đó làm đóng góp độc lập.

Sau khi xây chỉ mục có một tích phân 32 nút theo từng cặp và một lần tính trước giá trị mốc của mỗi ảnh. Công thức, rời rạc hóa và các đồng nhất thức phổ nằm trong [phân tích phép căn chỉnh](forced-pair-alignment.md). Các phiên bản và thời gian thực đo được lưu cùng kết quả.
