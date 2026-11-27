# Một chi tiết còn nhìn thấy giúp CLIP phân biệt gì?

Chưa có bằng chứng ổn định rằng ngữ cảnh mô tả che mất chi tiết trên **sketch**. Cùng một chi tiết lại có hiệu ứng rõ hơn trên phía **ảnh chụp** trong mẫu này. Một trạng thái phân biệt được hai vật thể cùng lớp cũng chưa đủ để sửa các lỗi truy hồi khác lớp của Semantic Dot.

## Thiết kế

Chọn theo băm với seed 42 tám lớp mỗi split Sketchy, tổng cộng 16 lớp khác nhau. Loại các vật thể đã kiểm tra trực quan ở pilot WordNet. Dùng 6 cặp sketch–ảnh nguồn đầu mỗi lớp để chọn một thuộc tính có hai trạng thái rõ ràng, chẳng hạn kéo mở/khép, cửa có vòm/đỉnh thẳng, cánh dơi mở/gấp. Khóa 16 cặp câu trước khi mở 10 vật thể xác nhận còn lại ở mỗi lớp. Trên bộ xác nhận, tôi chấm ảnh và sketch độc lập theo hai thứ tự ngẫu nhiên, ghi A, B hoặc chưa rõ. Tôi là người chấm duy nhất bằng AI. Commit nhãn đi trước commit chấm CLIP.

Trong 160 vật thể xác nhận, 137 sketch và 112 ảnh có trạng thái xác định. Có 88 cặp cùng trạng thái ở cả hai miền và 20 cặp có hai trạng thái đã ghi trái nhau. 12 lớp có đủ cả A và B sau khi ghép. 8 lớp có ít nhất hai vật thể mỗi trạng thái để thay ảnh nguồn bằng ảnh của một vật thể khác. Việc chọn thuộc tính thủ công chỉ dùng để kiểm tra mô hình.

Mỗi lần dùng CLIP ViT-B/32 gốc, hai câu giống nhau cho sketch và ảnh. AUC là xác suất biên điểm câu A trừ câu B của vật thể trạng thái A lớn hơn biên của vật thể trạng thái B trong cùng lớp. Hòa tính nửa điểm, các lớp có trọng số bằng nhau. Ngữ cảnh thật chứa mô tả chung cho cả hai trạng thái. Câu đệm trung tính được khớp độ dài chênh tối đa hai token BPE. [Giao thức](../../visible-attribute-context.md) · [Cặp câu](../../visible-attribute-predicates.tsv) · [Nhãn sketch](../../visible-attribute-sketch-labels.tsv) · [Nhãn ảnh](../../visible-attribute-photo-labels.tsv)

| Câu mô tả thuộc tính, AUC (%) trên 12 lớp đủ điều kiện | Sketch | Ảnh chụp |
|---|---:|---:|
| Chỉ nói về chi tiết | 77,21 | 69,19 |
| Thêm ngữ cảnh thật trước câu | 73,29 | 57,29 |
| Thêm ngữ cảnh thật sau câu | 83,81 | 64,25 |
| Thêm câu trung tính trước | 80,61 | 69,26 |
| Thêm câu trung tính sau | 85,34 | 68,61 |

Trung bình hai vị trí ngữ cảnh, chênh lệch so với chỉ nói chi tiết là **+1,34** điểm AUC trên sketch, khoảng bootstrap 95% theo lớp **[−11,98; +12,64]**. Trên ảnh là **−8,42 [−17,71; −1,30]**. So với thêm câu trung tính, kết quả là **−4,42 [−11,07; +1,43]** và **−8,17 [−15,73; −0,92]**. Có lớp chỉ có một ví dụ ở một phía, vì vậy cần đọc cả cỡ mẫu từng lớp. [Số đầy đủ và khoảng theo lớp](summary.json)

## Vì sao chọn đúng câu chưa chắc tìm được ảnh?

Ở 8 lớp đủ điều kiện, chọn hai ảnh khác vật thể có trạng thái đối nhau, loại ảnh nguồn của sketch khỏi phép truy hồi. Cung cấp **đúng câu ngắn** về trạng thái của sketch đạt 55,09% chính xác. Cho CLIP tự chọn câu từ sketch đạt 48,88%. Trên cùng các cặp ảnh, hai câu nói điều ngược nhau cùng ưu tiên **một bức ảnh** trong 83,58% trường hợp. Lựa chọn ảnh phụ thuộc mạnh vào phần phản hồi chung cho cả hai câu.

Tách điểm của hai câu thành phần chung và thành phần khác biệt. Khi biết trạng thái, dùng phần khác biệt đạt 64,20%, cao hơn 9,11 điểm với khoảng [−6,54; +23,23]. Khi CLIP phải tự xác định trạng thái từ sketch, kết quả còn **52,95%**, tăng **4,07 điểm phần trăm** với khoảng [−3,02; +15,27]. Đại lượng được đo là độ chính xác chọn giữa *hai* ảnh cùng lớp khi các cặp mô tả đối nghịch đã được cung cấp bằng tay. [Phép phân rã](../../visible-attribute-pair-decomposition.md) · [Kết quả](pair-terms.json)

## Kiểm tra trên chính đối thủ khác lớp của Semantic Dot

Với mỗi sketch, chọn ảnh đúng lớp cao hạng nhất và ảnh sai lớp cao hạng nhất từ **toàn gallery** theo Semantic Dot. Giữ nguyên hai ảnh cho mọi đối chứng. Có 76 truy vấn ở 12 lớp thỏa điều kiện chính. Semantic Dot chọn đúng 58 lần và sai 18 lần. Dùng riêng câu của trạng thái do người chấm cung cấp sửa 11 lỗi nhưng làm sai 16 trường hợp từng đúng. Dùng độ chênh giữa hai câu sửa 8 và làm sai 32. So với tráo trạng thái giữa các sketch cùng lớp, lợi thế trung bình theo lớp của trạng thái thật chỉ là +1,11 điểm [−2,13; +4,57] với một câu, và −3,20 [−13,84; +7,79] với độ chênh hai câu.

Trên 137 sketch có nhãn xác định, hiệu ứng so với tráo trạng thái cũng nhỏ, lần lượt +0,57 [−1,80; +2,89] và −0,96 [−8,39; +6,97]. Các phép tráo giữ nguyên lớp và tần số hai trạng thái. Khi mỗi câu sinh cùng thứ hạng toàn gallery cho một lớp, tráo nguyên các lựa chọn câu giữa những sketch cùng lớp bảo toàn tổng AP/P chính xác. Chỉ tiêu trên cặp đối thủ riêng của từng sketch kiểm tra một tính chất khác. Tệp JSON ghi riêng AP/P gốc của từng split trên nhóm truy vấn được phân tích. [Giao thức](../../visible-attribute-category-errors.md) · [Kết quả và mẫu số](category-errors.json)

## Novelty và quyết định

Hiện tượng phần nội dung chung chi phối so sánh cặp đã có prior art. Winoground dùng hai ảnh và hai câu kiểm tra khả năng ghép đúng. Một bản thảo tháng 11/2026 chứng minh hệ thức về phép loại phần chung cho image–caption. *Half-Truths* cho thấy chi tiết sai thêm vào câu có thể làm tăng điểm CLIP. Đóng góp của lần đo này là xác định được các giới hạn cụ thể trên sketch thật và truy ngược những lỗi khác lớp đã xảy ra. Những con số trên chưa ủng hộ việc phát triển riêng phép rút ngắn câu hoặc phép trừ hai caption thành method. [Winoground, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Thrush_Winoground_Probing_Vision_and_Language_Models_for_Visio-Linguistic_Compositionality_CVPR_2022_paper.pdf) · [Where Compositionality Hides, bản thảo](https://arxiv.org/abs/2611.30538) · [Half-Truths, NeurIPS 2026](https://arxiv.org/abs/2602.23906)

## Tái lập

Mã chọn ảnh ở `f1bec5e`, khóa mô tả ở `c2b7a52`, nhãn xác nhận ở `4a5ec4d`, chấm ở `2325dfa`, phép phân rã ở `b88deea` và đo đối thủ khác lớp ở `e8bf2b2`. Không dò cấu hình, không huấn luyện và không dùng gradient. Các phép đo chạy trên Kaggle L4. Hai split đều đã dùng để thăm dò trong quá trình phát triển trước đây. TU-Berlin ngoài phạm vi của chẩn đoán trực quan này.

Gói đầu chứa bảng điểm theo ảnh, các embedding câu, mã nguồn ảnh/sketch, thứ tự ngẫu nhiên và SHA-256 từng tệp. Gói tiếp theo chứa AP/P chính thức và ID đối thủ theo query. SHA-256 lần lượt là `71e911f100a1555e52ac4bf1da182c0477002f8145add1c7b2a13c8d49a844f8` và `f8a990d37584634438cf4e7afd4dc82dcaeac9556368418b11fcaa8cd4e68b46`. [Dữ liệu chấm thuộc tính](visible-attributes-L4.zip) · [Cặp cạnh tranh khác lớp](category-errors-L4.zip)
