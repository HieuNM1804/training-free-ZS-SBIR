# Tìm từ vựng nhỏ nhất giữ được các quyết định quan sát trên cả hai ảnh

## Hạn chế và giả thuyết

Dot trên 601 từ là xác suất hai lần rút tên độc lập cho đúng cùng một từ. Nó phụ thuộc vào cách chia tên trong từ điển. Nếu hai cách gọi có cùng hành vi khi CLIP đọc một vật thể nhưng sketch và ảnh ưu tiên hai cách gọi khác nhau, Dot có thể đánh mất phần khớp đó. Việc hai tên gần nhau trong không gian **text** chưa cho biết chúng có thay thế nhau trong một ảnh cụ thể hay không.

Trong lượt trước, bốn phép che đơn chỉ thu hẹp số cách ghép vùng hiệu dụng từ 24 xuống trung bình 23,80 / 23,77 / 23,87 trong top-K Dot. Đối chứng giữ các đáp ứng kép nhưng bỏ quan hệ vùng chỉ làm mAP thay đổi tối đa 0,04 điểm. Ở đây câu hỏi mới nằm bên trong từng ảnh: khi CLIP báo một từ, nó cung cấp những quyết định nào về phần bằng chứng thị giác đã bị bỏ? Một cặp tên chỉ được gộp nếu **cả hai ảnh** đều không phân biệt được chúng trong toàn bộ các quyết định đã định nghĩa. [Thực nghiệm đối sánh vùng](https://github.com/HieuNM1804/training-free-ZS-SBIR/tree/research/heldout-occlusion-correspondence)

## Một bài toán quyết định hữu hạn

Xét một ảnh $X$. Chọn ngẫu nhiên một trong hai vùng $i,j$ với xác suất bằng nhau, xóa token của vùng đã chọn trước block đầu tiên, rồi rút một tên $w$ từ phân bố từ của CLIP. Người quan sát chỉ nhận tên và được hỏi thao tác nào đã xảy ra. Với mất mát 0–1, tập đáp án Bayes tối ưu được xác định chính xác bởi
$$
s_{ij}^X(w)=\operatorname{sign}\!\left[p_X^{(-i)}(w)-p_X^{(-j)}(w)\right].
$$
Dấu bằng được lưu riêng. Lấy cả sáu câu hỏi từ bốn vùng can thiệp cùng kích thước. Vector sáu đáp án tạo trạng thái $s_X(w)$. Thay đổi cách đánh số vùng chỉ đổi đồng loạt các tọa độ hoặc dấu, giữ nguyên việc **hai từ có cùng trạng thái trong ảnh hay không**.

Đối với cặp ảnh $X,Y$, lập phân hoạch $\mathcal C_{XY}$ bằng cách nhóm những từ có cùng cặp $(s_X(w),s_Y(w))$. Đây là phân hoạch thô nhất mà từ tên nhóm có thể khôi phục toàn bộ sáu quyết định Bayes ở **từng** ảnh. Chứng minh gồm hai chiều: các quyết định là hàm của tên nhóm; gộp hai nhóm khác nhau làm mất ít nhất một quyết định ở ít nhất một ảnh. Đây là bảo đảm cho họ câu hỏi với prior đều đã nêu. Nó không khẳng định các từ trong cùng nhóm là từ đồng nghĩa trong ngôn ngữ hay hai vật thể cùng lớp.

Với posterior **ảnh đầy đủ** $q,p$, thu được xác suất hai lần rút tên nhất quán ở độ phân giải quan sát được
$$
S(X,Y)=\sum_{C\in\mathcal C_{XY}} q(C)\,p(C),\qquad q(C)=\sum_{w\in C}q(w).
$$
Khi một phía phân biệt hết mọi tên, $S=q^\top p$. Khi tách một tên thành nhiều mục có chung trạng thái và bảo toàn tổng khối lượng ở **mỗi** miền, $S$ bất biến kể cả lúc tỷ lệ chia khác nhau giữa hai miền. Đây là thao tác tái mã hóa trên phân bố. Thêm prompt trùng vào softmax thông thường có thể thay đổi cả tổng khối lượng. Giá trị $S$ cũng có thể tăng nếu ảnh quá mơ hồ và gộp các từ không liên quan. Do đó cần đối chứng bảo toàn kích thước từng nhóm.

Thiết kế chính dùng bốn phép xóa từng nhóm 3×3 patch góc của ViT-B/32, luôn giữ hàng và cột giữa. Không có bộ dò bộ phận, canny, graph hoặc ảnh hỗ trợ. Mỗi ảnh được mã hóa độc lập, hình học giữa hai ảnh không được gán tương ứng. Các ảnh sketch dùng hai lần đọc gốc và phản xạ như Dot; can thiệp cùng vị trí vật lý trước khi lấy trung bình. Các ảnh kép đã tính chỉ phục vụ kiểm định và ablation. Thuật toán chính dùng một lượt nguyên vẹn và bốn lượt xóa cho mỗi góc nhìn.

## Đối chứng đã định trước

1. **Cùng cơ hội gộp tên.** Giữ nguyên kích thước *từng* nhóm chung và mọi xác suất trong hai ảnh. Gán ngẫu nhiên lại tên vào phân hoạch bằng cùng một hoán vị đều. Kỳ vọng chính xác là $S_{\rm size}=D+(1-D)\sum_C |C|(|C|-1)/(V(V-1))$, với $D=q^\top p$, $V=601$. So hai nhánh ở cùng số ảnh mới vào top-K kiểm tra giá trị riêng của danh tính các từ được gộp.
2. **Trạng thái của ảnh tương tự.** Đổi bảng trạng thái giữa các ảnh RGB có cùng tên đứng đầu và gần nhau về độ tập trung posterior. Giữ nguyên posterior của từng ứng viên. Đây là đối chứng cho cấu trúc từ vựng dùng chung trong các ảnh có ngữ nghĩa giống nhau. Suy luận chính vẫn độc lập từng cặp.
3. **Giá trị của yêu cầu cả hai ảnh.** Chỉ dùng phân hoạch của một phía. Một ablation khác yêu cầu trùng thêm 12 quyết định khi vùng thứ ba bị xóa trước. Họ câu hỏi và cách xử lý hòa được khóa trước khi xem kết quả.
4. **Dự đoán trong ngữ cảnh mới.** Trên 512 ảnh chọn theo seed của mỗi miền mỗi split, lấy 32 tên đầu theo posterior nguyên vẹn. Với cặp được gộp, tìm từ đối chứng trong cùng dải hạng ở cùng ảnh, giữ cùng từ thứ nhất và ghép theo cosine text gần nhất. Đo tỷ lệ hai tên còn trả cùng đáp án ở 12 câu hỏi có vùng thứ ba bị xóa. Mỗi điều kiện dùng cùng số cặp. Báo tỷ lệ có đối chứng và chênh lệch cosine còn lại.
5. **Chấm truy hồi.** Cấu hình chính được khóa trước cả ba split. Không chọn tham số trên unseen trong lượt này. Dùng toàn bộ hai split Sketchy và TU-Berlin 30 lớp với evaluator KD-SBIR gốc, CLIP ViT-B/32 và seed 42. Chỉ nhãn sau xếp hạng mới được dùng. Lợi ích về khả năng dự báo ngữ cảnh và độ chính xác khi chọn ảnh mới là hai phát biểu độc lập. Để bảo vệ đóng góp về retrieval, cần bằng chứng còn có lợi khi giữ cố định cơ hội gộp và khi so với bảng từ của một ảnh tương tự.

## Thuật toán và prior art

Với từng cặp, sắp 601 từ theo khóa kép $(s_X,s_Y)$ và cộng hai khối lượng trong từng đoạn liên tiếp. Chi phí là $O(V\log V)$ thời gian và $O(V)$ bộ nhớ tạm. Với một lời gọi retrieval lớn, phép khai triển $S=D+\sum_{u<v} (q_u p_v+q_v p_u)\,\mathbf1[s_X(u)=s_X(v)]\,\mathbf1[s_Y(u)=s_Y(v)]$ cho phép dùng nhân ma trận trên GPU, chia khối các cặp từ để chấm chính xác đủ gallery. Mã đánh giá so kết quả với phương pháp cộng khối lượng trực tiếp bằng FP64. Chi phí thực phải được báo cùng kết quả.

QBSE đã dùng phân bố concept để biểu diễn ảnh. Visual Word Ambiguity đã mô hình hóa việc đặc trưng có thể gắn cho nhiều mục trong codebook. Câu hỏi của thí nghiệm này là có thể **nhận dạng được sự phân biệt nào** giữa các tên trong một cặp ảnh cụ thể. [QBSE, IEEE TMM](https://doi.org/10.1109/TMM.2007.900138) · [Visual Word Ambiguity, TPAMI](https://ivi.fnwi.uva.nl/isis/publications/2010/vanGemertTPAMI2010/vanGemertTPAMI2010.pdf)

Trừu tượng hóa trạng thái có bảo toàn quyết định, tích giữa các phân bố và chỉ báo hai phần tử cùng nhóm đều có tiền lệ. Phần nghiên cứu cụ thể là thiết kế thí nghiệm quan sát cho một tên CLIP, tìm từ vựng chung tối thiểu cho **hai** thí nghiệm, chứng minh bất biến với tách tên theo miền và kiểm tra các cặp từ trên can thiệp chưa dùng. Không thể khẳng định một phép gộp đại số là phát minh mới. [Li, Walsh & Littman, 2006](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/camera-ready-9.pdf) · [Jebara et al., JMLR 2004](https://www.jmlr.org/papers/volume5/jebara04a/jebara04a.pdf) · [Davies & Ghahramani, 2014](https://arxiv.org/abs/1402.4293)

Nghiên cứu về các tập patch đủ cho quyết định kiểm tra đầu ra dưới phép che. Ở đây kết quả được dùng để định nghĩa khi nào hai tên từ vựng là hai quan sát khác nhau trong *một* ảnh, sau đó mới đối chiếu hai ảnh. Các phương pháp prompt learning trong ZS-SBIR, dịch ảnh qua tên của FreeDom, hợp nhất tín hiệu của BASIC và căn chỉnh caption bằng transport của STiTch trả lời các câu hỏi khác. Tối đa ba nguồn trực tiếp cho thiết kế là QBSE, trừu tượng hóa bảo toàn quyết định và can thiệp bằng patch. [Jiang et al., CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Jiang_Comparing_the_Decision-Making_Mechanisms_by_Transformers_and_CNNs_via_Explanation_CVPR_2024_paper.html) · [SpLIP, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/3426_ECCV_2024_paper.php) · [FreeDom, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/papers/Efthymiadis_Composed_Image_Retrieval_for_Training-Free_Domain_Conversion_WACV_2025_paper.pdf) · [BASIC, NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/hash/6391a02f02dc529cfa3f242fb4343c6a-Abstract-Conference.html) · [STiTch, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Li_STiTch_Semantic_Transition_and_Transportation_in_Collaboration_for_Training-Free_Zero-Shot_CVPR_2026_paper.html)
