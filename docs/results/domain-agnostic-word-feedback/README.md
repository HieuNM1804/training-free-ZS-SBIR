# Phản hồi theo cùng một khái niệm cho truy hồi giữa hai miền ảnh

**Đối tượng nghiên cứu** là một truy vấn ảnh và một gallery đích không nhãn. Câu hỏi là khi truy vấn và ứng viên cùng gợi tới một khái niệm, liệu các ảnh đích khác gợi tới khái niệm đó có củng cố cách hiểu này không. Cùng một hàm nhận embedding ảnh nguồn, embedding gallery đích, một bộ embedding từ công khai và nhiệt độ CLIP. Hàm không cần tên miền, nhãn lớp, sketch khác, tên benchmark hay tham số chọn riêng cho một bộ. Giả định là hai miền còn có tín hiệu ngữ nghĩa chung trong CLIP và gallery có ảnh có thể cung cấp bằng chứng bổ sung. Từ vựng Open Images và thiên lệch của CLIP vẫn có thể ảnh hưởng đến kết quả.

## Cơ chế

- Mã hóa cùng 20.931 tên công khai thành đúng một bộ vector CLIP ViT-B/32. Đưa tên từ nguyên văn cho bộ mã hóa văn bản ở cả hai phía.
- Dùng xác suất từ của CLIP để mô tả mỗi ảnh đích. Gộp mô tả trên mọi ảnh đích còn lại theo cosine giữa ảnh ở nhiệt độ đã có trong checkpoint. Không chọn top-K hoặc bán kính trên dữ liệu. Gọi mô tả sau bước này là $B_i(w)$.
- Với query bất kỳ, gọi likelihood của từ là $l(w)$ và điểm tuyến tính đầu tiên của ảnh $j$ là $e_j=\sum_w B_j(w)l(w)$. Với mỗi từ đang giải thích ứng viên $i$, lấy bằng chứng trung bình của các ảnh đích khác ủng hộ chính từ đó. Đặt $h_i(w)=\sum_{j\ne i}B_j(w)e_j/\sum_{j\ne i}B_j(w)$. Khi không có ảnh nào khác hỗ trợ, bằng chứng bổ sung của từ bằng 0.
- Điểm cuối là $\mathbb{E}_{w\sim B_i}[l(w)h_i(w)]$. Đối chứng cho hai tín hiệu chọn từ độc lập là $\mathbb{E}_{B_i}[l]\,\mathbb{E}_{B_i}[h_i]$. Chênh lệch giữa **điểm số** của hai cách là hiệp phương sai của hai nguồn bằng chứng trên phân bố từ của ứng viên. Chênh lệch mAP phải đo thực tế. Cả hai cách dùng cùng ảnh, cùng từ, cùng trọng số CLIP và cùng quy tắc loại ảnh ứng viên.

Cơ chế tự đề xuất nằm ở việc xác nhận bằng chứng theo cùng khái niệm và khả năng xác định chính xác ảnh/từ nào giải thích sự đảo hạng qua hai lượt sử dụng truy vấn. FreeDom cho tiền lệ chuyển ảnh qua từ vựng để truy hồi xuyên miền. Relevance models dùng bộ sưu tập để ước lượng phân bố từ liên quan. Logit scale và các embedding lấy từ CLIP gốc. [FreeDom, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Efthymiadis_Composed_Image_Retrieval_for_Training-Free_Domain_Conversion_WACV_2025_paper.html) · [Lavrenko và Croft, SIGIR 2001](https://www.inf.ed.ac.uk/publications/report/1190.html) · [CLIP, ICML 2021](https://proceedings.mlr.press/v139/radford21a.html)

## Kết quả không chọn tham số theo benchmark

Cấu hình chính đã cố định ở commit f67cdd4 trước khi xem bất kỳ số nào của lượt chạy này. Các kết quả lịch sử trên hai benchmark đã có sẵn. Ext-1 dùng mAP@all / P@100, Ext-2 dùng mAP@200 / P@200 và TU-Berlin 30 lớp dùng mAP@all / P@100. Mỗi ô là phần trăm.

| Cấu hình | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin |
|---|---:|---:|---:|
| Cùng word bank, graph ở nhiệt độ CLIP, chấm tuyến tính | 76,20 / 78,10 | 71,17 / 69,88 | 75,06 / 76,64 |
| Hai từ độc lập, cùng word bank và graph | 78,62 / 79,19 | 72,03 / 70,74 | 77,41 / 77,67 |
| Cùng một từ, cùng word bank và graph — cấu hình chính | 78,03 / 78,49 | 71,18 / 70,14 | 76,94 / 76,94 |
| Cùng một từ, bỏ propagation RGB | 76,22 / 77,88 | 70,35 / 69,50 | 74,02 / 76,15 |
| Cùng một từ, dùng graph từng chọn trên Sketchy | 81,76 / 80,60 | 72,72 / 71,37 | 79,62 / 79,54 |
| Cùng một từ, thêm prompt sketch/photo | 79,37 / 79,84 | 72,38 / 71,27 | 78,15 / 77,96 |
| Cùng một từ, thêm prompt sketch/photo và graph cũ | 82,56 / 81,46 | 74,07 / 72,73 | 79,77 / 80,62 |

So với chấm tuyến tính, ứng viên cùng một từ tăng 1,83, 0,01 và 1,88 điểm mAP. Đối chứng hai từ độc lập vượt ứng viên trên cả ba split. Mức chênh của cấu hình cùng từ so với độc lập là −0,59 (CI 95% theo lớp từ −1,06 đến −0,11), −0,85 (từ −1,44 đến −0,25), và −0,47 (từ −0,92 đến −0,03). Đây là bằng chứng bất lợi cho việc áp đặt một từ chung khi bỏ thông tin về tên miền.

Trung vị số láng giềng hiệu dụng, tính bằng nghịch đảo tổng bình phương các xác suất chuyển, chỉ là 3,07, 2,96 và 6,12. Lần lượt 40,7%, 42,5% và 18,1% ảnh nhận ít hơn hai láng giềng hiệu dụng. Gallery không nhãn đã cho thấy nhiệt độ CLIP khá sắc khi dùng cho quan hệ ảnh–ảnh. Graph lịch sử giúp mAP khi giữ từ nguyên văn; kết quả này chưa phân tách các tác động của độ rộng, toán tử và số lần lan truyền.

## Phạm vi của chứng nhận

Đối với hai ảnh bất kỳ, điểm chênh có dạng $l^\top H l$. Sau khi vô hiệu hóa tọa độ $k$ ở phía truy vấn, tính cả tác động trực tiếp lẫn tác động gián tiếp qua mọi ảnh cung cấp phản hồi. API trả về chênh lệch sau từng phép bỏ từ và từ có khả năng đảo cặp mạnh nhất, dùng hai lượt quét gallery cho mỗi ứng viên trong FP64. Chứng nhận áp dụng cho phép can thiệp vào likelihood truy vấn; lớp dự đoán, nội dung điểm ảnh và độ tin cậy ngữ nghĩa là các đối tượng đo khác.

Kiểm tra đủ 20.931 từ cho 6 truy vấn định trước với 72 ảnh gallery ở mỗi fixture, tổng cộng 125.586 phép xóa thực tế. Có 0 trường hợp sai dấu ngoài ngưỡng sai số và sai lệch lớn nhất theo điểm ban đầu là 1,08 × 10⁻⁶. Kiểm tra API riêng trên 11 truy vấn và 62 ảnh cho sai khác lớn nhất theo từng phép về thứ tự gallery, xử lý từng truy vấn và đổi đồng thời cơ sở tọa độ dưới 2,5 × 10⁻⁵. Những kiểm tra này xác nhận công thức và mã trên các trường hợp đã thử.

## Ý nghĩa cho hướng bài báo

Nên giữ hai nguồn bằng chứng và phép can thiệp có thể kiểm tra làm nền nghiên cứu. Chỉ số cao của cấu hình lịch sử không chứng minh rằng ràng buộc dùng chung một từ sẽ chuyển giao xuyên miền. Hướng tiếp theo phát sinh từ đối chứng là cho hai từ biểu diễn hai khía cạnh bổ sung và kiểm soát chính xác phân bố của từng nguồn. [Thiết kế kiểm tra đã khóa trên nhánh riêng](https://github.com/HieuNM1804/training-free-ZS-SBIR/blob/5ea27f4/docs/COMPLEMENTARY_WITNESS_PROTOCOL.md)

Phản hồi qua từ, mở rộng truy vấn bằng đồng xuất hiện và chứng nhận thứ hạng đều có tiền lệ. Phần cần đối chiếu kỹ với các công trình đó là luật chọn ảnh xác nhận theo khái niệm đang đóng góp cho từng ứng viên và khả năng quy trách nhiệm chính xác cho một thay đổi của thứ hạng khi toàn bộ phản hồi được tính lại. Cơ chế và phép thử này dùng được cho các cặp miền khác có tín hiệu từ chung. Dữ liệu đã duyệt chỉ kiểm chứng được hai benchmark của cùng chiều sketch → photo. [ICFRR](https://arxiv.org/html/2303.17703v2) · [Mở rộng truy vấn ảnh trong không gian khái niệm](https://pubmed.ncbi.nlm.nih.gov/21822350/) · [CertDR](https://doi.org/10.1145/3511808.3557256) · [Chứng nhận truy hồi ảnh, WACV 2023](https://openaccess.thecvf.com/content/WACV2023/html/Kakizaki_Certified_Defense_for_Content_Based_Image_Retrieval_WACV_2023_paper.html)

Mọi phép chấm xếp hạng dùng toàn bộ 29.003 truy vấn trên Colab L4 và hàm KD-SBIR gốc. [Mã tính điểm và giải thích](../../../tf_sbir/domain_neutral_witness.py) · [Thiết kế đã khóa](../../DOMAIN_NEUTRAL_PROTOCOL.md) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU](tuberlin.json) · [Khóa](freeze.json) · [Kiểm tra API](api-check.json) · [Kiểm tra giải thích](explanation-check.json) · [Dữ liệu từng truy vấn](per-query.zip)
