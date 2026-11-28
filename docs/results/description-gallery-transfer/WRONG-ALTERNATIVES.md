# Khi CLIP phải chọn sai, sketch và ảnh cùng lớp có chọn cùng đáp án?

**Có, kể cả khi ảnh chụp là một vật thể khác.** Trên những cặp lớp dễ nhầm, xác suất sketch và ảnh chọn trùng đối thủ thị giác cao hơn cặp khác lớp **6,58 điểm phần trăm ở Ext-1 và 9,00 ở Ext-2**. Chênh lệch còn **4,48 / 7,45 điểm** khi giữ cố định tên sai được chọn. Đây là chẩn đoán về câu trả lời của CLIP. Chỉ số đo ở đây là xác suất trùng đáp án. mAP cần được đo bằng thứ hạng trên gallery.

## Cách lập đối chứng

Với hai lớp A và B, bỏ cả A và B khỏi các đáp án cho mọi vật thể. Đo mức trùng câu trả lời ở hai cách ghép cùng lớp A–A, B–B và hai cách ghép khác lớp A–B, B–A. Chia đều trọng số cho hai chiều. Với cùng một cặp lớp, cả bốn cách ghép thấy đúng cùng một danh sách đáp án. Đây là cách loại khả năng đoán lớp chỉ dựa vào đáp án nào bị ẩn.

Mỗi sketch dùng embedding của hình gốc và hình lật như trong phép đo thị giác trước. Một vật thể nguồn có nhiều bản vẽ vẫn có tổng trọng số bằng các vật thể khác. Ảnh chụp để so sánh nằm trong tập ảnh chưa từng làm nguồn vẽ, với 14.600 ảnh ở Ext-1 và 10.452 ở Ext-2. Dùng đủ 14.179 / 12.694 sketch. Mỗi fold dùng 50 vật thể khác mỗi lớp để lập trọng tâm RGB, chọn tên công khai theo AP trên ảnh tham chiếu và xác định lớp dễ nhầm. Nhãn chỉ tham gia chẩn đoán. Tên gán trùng được gộp trước khi chọn. Các danh sách ngắn này cần thông tin lớp và khác với toàn bộ 601 tên của Semantic Dot. [Giao thức](../../native-wrong-alternatives.md) · [Kết quả Ext-1](rivals-summary-sketchy_1.json) · [Ext-2](rivals-summary-sketchy_2.json)

| Đáp án dành cho sketch và ảnh, trên các cặp khó | Cùng lớp Ext-1 | Khác lớp Ext-1 | Chênh Ext-1 | Cùng lớp Ext-2 | Khác lớp Ext-2 | Chênh Ext-2 |
|---|---:|---:|---:|---:|---:|---:|
| Đại diện RGB được chọn đầu tiên | 31,23 | 24,65 | +6,58 | 41,25 | 32,24 | +9,00 |
| Tên công khai được chọn đầu tiên | 26,69 | 16,67 | +10,02 | 31,32 | 15,70 | +15,62 |
| Thứ tự của một cặp đại diện RGB lấy đều | 74,64 | 70,18 | +4,46 | 75,46 | 70,75 | +4,71 |
| Thứ tự của một cặp tên công khai lấy đều | 64,10 | 57,67 | +6,43 | 62,26 | 56,47 | +5,80 |

Khoảng bootstrap 95% theo lớp cho hàng đầu là [4,02, 9,36] / [4,46, 13,96]. Tính đều trên **tất cả** lớp đối thủ thay cho đối thủ khó, chênh lệch top-1 RGB là +16,55 / +21,22. Khi loại bốn lớp chung của hai split khỏi cả truy vấn, đối thủ và đáp án, chênh lệch trên cặp khó còn +9,97 / +13,51. Với tên công khai, chênh lệch trên các lớp riêng là +8,27 / +15,28. Thứ tự tương đối giữa hai cách đọc thay đổi trên Ext-1 khi thay danh sách lớp.

## Giữ cố định đáp án ở kênh kia

Trong từng cặp A, B, chỉ so sánh những đáp án xuất hiện ở cả bốn nhóm gồm sketch A, sketch B, ảnh A và ảnh B. Trong từng nhóm như vậy, tính lại cùng chênh lệch xác suất. Gán trọng số theo khối lượng nhỏ nhất của bốn phân bố để chúng có cùng cơ cấu đáp án. “Phần chung” là tổng các trọng số ấy trước khi chuẩn hóa. Nó biểu thị khối lượng mà bốn phân bố có thể cùng đại diện. [Giao thức](../../native-rival-conditional.md) · [Ext-1](rival-conditional-sketchy_1.json) · [Ext-2](rival-conditional-sketchy_2.json)

| Tín hiệu đo trên cặp khó | Chênh Ext-1 [95%] | Phần chung Ext-1 | Chênh Ext-2 [95%] | Phần chung Ext-2 |
|---|---:|---:|---:|---:|
| Đại diện RGB khi giữ nguyên tên sai | +4,48 [2,51, 6,71] | 18,48% | +7,45 [3,86, 11,43] | 15,64% |
| Tên công khai khi giữ nguyên đại diện RGB sai | +8,91 [4,93, 13,20] | 24,71% | +16,59 [10,05, 23,81] | 24,26% |

Các ước lượng trên cho trọng số bằng nhau cho mỗi lớp. Khi cho trọng số theo phần chung, hai chênh lệch trên Ext-2 lần lượt là **+3,02 và +10,15**, cho thấy sự phụ thuộc đáng kể vào nhóm vật thể. Chẩn đoán chỉ giữ cố định **một đáp án top-1**. Vector điểm đầy đủ của cùng kênh có thể vẫn chứa thông tin giải thích phần chênh lệch. Một phương pháp khai thác hiện tượng này cần sửa được các cặp bị baseline xếp sai và cải thiện mAP. So sánh cấu trúc biểu diễn hoặc mẫu lỗi giữa ảnh và bản vẽ đã có tiền lệ. [Fan et al., Cognitive Science](https://pubmed.ncbi.nlm.nih.gov/30125986/) · [Rajalingham et al., Journal of Neuroscience](https://pubmed.ncbi.nlm.nih.gov/30006365/)

Chạy trên Kaggle L4, OpenAI CLIP ViT-B/32 đóng băng, seed 42. Không dò tham số. Mã chấm đầu tiên ở `5c96f61`, phép đo có điều kiện ở `ab19ed7`. Với ba cách đọc và một cặp lớp cố định trên mỗi fold, liệt kê trực tiếp mọi cặp vật thể tái lập xác suất từ bảng tần suất với sai số tối đa 2,23 × 10⁻¹⁶. Bảng kết hợp tên và RGB tái lập tần suất biên của mọi cặp khó với sai số tối đa 5,56 × 10⁻¹⁶. Gói gốc SHA-256 `6da366673369814b3fd6827a914946a94a55c0e089ee6a5194e0609d4d4234eb`. Gói có điều kiện SHA-256 `05abf7498f0ab30df4ae7da7416d0dcb25e5d352af6048cc538a5afe3c0a17e3`. [Tensor và ID](rivals-L4.zip) · [Bảng có điều kiện](conditional-rivals-L4.zip)
