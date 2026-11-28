# CLIP có nhận ra hai vật thể cùng loại nếu không được cho tên lớp?

CLIP nhận ra cách ghép đúng khá tốt khi so hai cặp bằng cosine thị giác gốc. Nhưng khi đặt hai hình cạnh nhau rồi hỏi liệu hai vật thể có cùng loại, kết quả thiếu ổn định giữa hai split. Câu hỏi duy nhất đã khóa chưa đủ cơ sở để xây bộ truy hồi. Một thí nghiệm riêng trên lỗi thực cho thấy khả năng ghép đúng hai truy vấn cũng không đồng nghĩa với khả năng sửa mọi thứ hạng bằng một hằng số cho mỗi ảnh.

## Cách ghép hai sketch với hai ảnh

Chọn hai lớp A, B. Mỗi lớp cung cấp một sketch và một ảnh của **vật thể khác ảnh nguồn**. So cách ghép hai cặp cùng lớp với cách ghép chéo. Mỗi hình xuất hiện đúng một lần trong mỗi phương án, nên mọi thành phần cộng riêng theo từng hình triệt tiêu khi lấy hiệu. Trên bảng 224×224, CLIP đọc hai hình vuông 112×112 cạnh nhau. Đọc cả hai chiều trái/phải. Dùng chênh giữa hai câu cố định `two objects of the same kind.` và `two objects of different kinds.`. Thử thêm cặp mà hai ảnh đều tự đề cử đúng cùng danh từ. Tên chỉ được dùng để chọn đối chứng. [Giao thức, nguồn và đối chứng](../../visual-category-relation.md)

Ext-1 lấy 1.200 lần thử bình thường, gồm bốn lần cho mỗi một trong 300 cặp lớp. Điều kiện cùng tên có 1.172 lần thử, bao phủ 293 cặp lớp. Ext-2 có tương ứng 840 lần thử trên 210 cặp và 768 lần thử trên 192 cặp. Hai split có chồng lấp. Báo cáo độ chính xác ghép cặp theo %, mỗi cặp lớp có trọng số bằng nhau.

| Cách quyết định | Ext-1 thường | Ext-1 cùng tên | Ext-2 thường | Ext-2 cùng tên |
|---|---:|---:|---:|---:|
| Cosine CLIP gốc, cộng điểm của hai cặp | 98,17 | 94,80 | 96,55 | 91,93 |
| CLIP đọc câu hỏi cùng loại trên hai hình đặt cạnh nhau | **60,58** | **57,59** | **54,17** | **52,60** |
| Cùng câu hỏi, thay sketch bằng ảnh nguồn thực | 60,25 | 53,75 | 54,76 | 52,34 |
| Cùng câu hỏi, thay sketch bằng chính bản sao của ảnh ứng viên | 53,83 | 49,23 | 37,14 | 38,28 |

Ở điều kiện cùng tên, khoảng bootstrap 95% theo lớp của cách đọc câu hỏi là **[50,99 đến 64,40]** và **[43,90 đến 61,13]**. Trên Ext-2, ngay cả khi hai khung trong cách ghép đúng chứa cùng một ảnh, cách đọc này vẫn kém ngẫu nhiên. Vì vậy có thể kết luận về giới hạn của **cặp prompt và cách đặt hình đã thử**. Không thể suy rộng thành CLIP thiếu mọi thông tin về quan hệ. Cosine trực tiếp đã cho thấy thông tin phân biệt trong biểu diễn ở chính những cặp này. Đối chứng cộng biểu diễn của hai nửa rồi chuẩn hóa chỉ đạt 12,58 / 8,69% trong điều kiện thường, nên vượt đối chứng ấy tự nó là một tiêu chí yếu. [Kết quả sketch Ext-1](summary-sketchy_1.json) · [Ext-2](summary-sketchy_2.json) · [Hai đối chứng chuyển miền và lặp ảnh](relation-controls-L4.zip)

Các bài về ghép ảnh–câu và suy luận quan hệ gợi ý cách giữ nguyên thành phần khi đổi cặp. Các nghiên cứu đó không bảo đảm rằng một câu hỏi tự nhiên sẽ đọc được quan hệ cùng loại của hai hình trong CLIP B/32. [Winoground, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Thrush_Winoground_Probing_Vision_and_Language_Models_for_Visio-Linguistic_Compositionality_CVPR_2022_paper.html) · [Beyond the Doors of Perception, NeurIPS 2024](https://papers.neurips.cc/paper_files/paper/2024/hash/ed99503856b2ba85092e5413add65d86-Abstract-Conference.html)

## Một độ lệch cố định có sửa được các lỗi ban đầu không?

Trên **mọi sketch**, giữ ảnh đúng lớp có điểm cao nhất sau khi loại ảnh nguồn và ảnh sai lớp đứng đầu. Chỉ lấy các trường hợp ảnh sai thắng nghiêm ngặt. Nếu ảnh B lấn ảnh đúng A ở query thứ nhất, việc sửa yêu cầu tăng điểm A so với B. Nếu các yêu cầu tương tự tạo thành vòng A → B → C → A, tổng phần điểm phải tăng dương trong khi tổng các hằng số quanh vòng bằng 0. Ít nhất một yêu cầu trong vòng phải thất bại. Chương trình lưu từng chu trình, ID các query chứng minh nó và một thứ tự khả thi cung cấp cận trên. Đây là bài kiểm tra khả năng hiệu chỉnh trên cặp cố định. [Định nghĩa và cách kiểm chứng](../../visual-gallery-bias-cycles.md)

Bảng ghi **cận dưới – cận trên cho số lỗi tối thiểu buộc phải còn** đối với từng họ hằng số. Cận dưới đếm chu trình có các cạnh truy vấn rời nhau. Cận trên là số bất đẳng thức thực sự bị vi phạm với một thứ tự cụ thể. Mỗi dòng có thể dùng thông tin nhãn khi chẩn đoán.

| Điểm gốc | Split | Cặp sai ban đầu | Một hằng số / ảnh | Một hằng số / tên công khai | Một hằng số / lớp thật |
|---|---|---:|---:|---:|---:|
| Cosine thị giác, một lượt | Ext-1 | 8.982 | 9–11 | 396–468 | 18–23 |
| Cosine thị giác, một lượt | Ext-2 | 8.018 | 28–51 | 558–683 | 65–75 |
| Semantic Dot | Ext-1 | 4.235 | **84–173** | **510–801** | 643–800 |
| Semantic Dot | Ext-2 | 4.463 | **157–274** | **1.017–1.398** | 797–929 |

Với Semantic Dot, chỉ riêng các chu trình đã chứng minh mọi cách cộng hằng số theo tên đều phải bỏ lại **ít nhất 12,04% / 22,79%** số cặp sai đã chọn. Cận thấp hơn nhiều khi cấp một hằng số tự do cho từng ảnh. Lúc đó có thể sửa phần lớn *cặp sai cũ* bằng một thứ tự sử dụng nhãn và được chọn trên chính các lỗi ấy. Thứ tự này có thể đẩy nhiều ảnh sai khác lên hoặc làm hỏng các query vốn đúng. Các cận không phải mAP, không ước tính hiệu quả ngoài mẫu và không chứng minh hiệu chỉnh hằng số sẽ làm benchmark tốt lên. [Chứng nhận Ext-1](cycles-sketchy_1.json) · [Ext-2](cycles-sketchy_2.json)

Một hệ quả cho việc chọn ý tưởng tiếp theo là kiểm tra xem một phép ghép trông có vẻ mới có thực sự dùng thêm thông tin tùy query. Chẳng hạn, cho một query đang xét cùng một ngân hàng truy vấn phụ cố định tham gia bài toán ghép tổng điểm. Khi buộc query hiện tại nhận ảnh g, tổng tối ưu bằng điểm của cặp (q,g) cộng với giá trị tối ưu để ghép các truy vấn phụ vào gallery còn lại. Phần thứ hai chỉ phụ thuộc g nếu các truy vấn phụ và tập ảnh cố định. Theo đúng các giả định đó, xếp hạng bằng phép ghép tương đương cộng một hằng số cho mỗi ảnh. Chỉ thêm bài toán ghép này chưa tạo ra cơ chế tương tác mới. Các mô hình chuẩn hóa bằng ngân hàng truy vấn cũng là đối chứng liên quan. [The Assignment Game](https://doi.org/10.1007/BF01753437) · [Querybank Normalisation, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Bogolin_Cross_Modal_Retrieval_With_Querybank_Normalisation_CVPR_2022_paper.pdf)

## Tái lập

Kaggle 4×L4, OpenAI CLIP ViT-B/32, seed 42. Mã đọc quan hệ ở `c94cab74811011062a38079584257aa9a1bd6ae3`, đối chứng cùng miền/lặp ảnh ở `a9d8aa0fb66fdf14b67443cffca1f240f74c4cb1`, chương trình dựng và kiểm tra chu trình ở `022db06adc49b28b36497b68f06827250a206cad`. Dữ liệu khác cùng lớp và các sketch khác chỉ phục vụ chẩn đoán sau khi định nghĩa điểm. Không có lần chấm TU hay thay đổi tham số.

| Gói kết quả | SHA-256 |
|---|---|
| [Cặp ảnh thực và biên ghép](relation-L4.zip) | `50cb5311a020dcbd0c14d89dcec67179673036e19efb4ca61af8256689c7e92e` |
| [Đối chứng ảnh nguồn và ảnh lặp](relation-controls-L4.zip) | `aff612309ef1d7a434a3bd814f64338846b3bdc6d4ebaa629568a70bf6ced2a9` |
| [Chứng nhận chu trình và từng cặp sai](offset-cycles-L4.zip) | `c7d77a814b74b61fb2d73b5f966613ae0c9612c1c6eaa45dca88d9305f4d8c6e` |
