# Ảnh làm giảm phản ứng của CLIP với tên in cạnh nó, nhưng hiệu này truy hồi kém

**Quyết định dừng phương pháp.** Chỉ mục xây từ độ suy giảm của phản ứng với chữ đạt **13,44 / 19,85** trên Sketchy Ext-1 và **16,95 / 15,49** trên Ext-2. Semantic Dot đạt **75,35 / 77,10** và **72,66 / 71,07**. Chênh lệch chính thức là −61,91 và −55,72 điểm mAP. Các vector tương tác đều khác 0, nhưng không cho thấy lợi thế ổn định khi phân biệt hai ảnh có cùng tên và gần bằng nhau về cosine thị giác. [Giao thức khóa trước](../../photo-name-interaction.md)

## Thiết kế

Mỗi ảnh tự đề cử một nghĩa trong 1.135 định nghĩa WordNet ẩn tên. In danh từ công khai ứng với nghĩa đó trong một dải 32 pixel, đặt bên dưới ảnh. Giữ nguyên kích thước ảnh và vị trí dải chữ trong cả bốn điều kiện. Đọc token CLS cuối của CLIP qua phép chiếu tuyến tính gốc trước bước chuẩn hóa cuối. Vector chỉ mục bằng **ảnh hưởng của việc thêm chữ vào ô trống trừ đi ảnh hưởng của việc thêm cùng chữ vào ảnh thật**. Nếu encoder chỉ cộng độc lập tín hiệu ảnh và chữ, hiệu này bằng 0. Một sketch dùng embedding CLIP gốc của một lượt đọc để truy vấn toàn gallery.

Mỗi ảnh cần ba lượt mã hóa cho phương pháp chính, gồm đề cử danh từ, đọc khung ảnh không chữ và đọc ảnh kèm chữ. Các thẻ chữ đứng riêng được tính dùng chung. Mỗi ảnh lưu một vector 512 chiều. Việc in chữ trong ảnh đã có tiền lệ ở LoGoPrompt, và CLIP có khả năng mã hóa cả chữ viết lẫn vật thể. Giả thuyết cụ thể ở đây là phần phản ứng chữ bị suy giảm có thể đại diện cho nội dung mà ảnh đã thể hiện và sketch cũng nhận ra. [LoGoPrompt, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Shi_LoGoPrompt_Synthetic_Text_Images_Can_Be_Good_Visual_Prompts_for_Vision-Language_Models_ICCV_2023_paper.html) · [Materzyńska et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Materzynska_Disentangling_Visual_and_Written_Concepts_in_CLIP_CVPR_2022_paper.html)

## Truy hồi trên toàn tập unseen

OpenAI CLIP ViT-B/32 đóng băng, Kaggle 4×L4, seed 42. Ext-1 có 14.179 sketch và 17.100 ảnh, dùng mAP@all / P@100. Ext-2 có 12.694 sketch và 12.552 ảnh, dùng mAP@200 / P@200. Mỗi hàng thử nghiệm mới gọi trực tiếp hàm KD-SBIR gốc trên các embedding thực, theo đúng thứ tự gallery. Semantic Dot là kết quả từng truy vấn đã lưu của baseline, có lượt nhìn sketch gốc và lật như đã thống nhất. Một cấu hình được khóa trước khi chấm. TU-Berlin chưa được dùng để đánh giá phương pháp này.

| Vector dùng để lập chỉ mục ảnh | Ext-1 | Ext-2 |
|---|---:|---:|
| Semantic Dot | **75,35 / 77,10** | **72,66 / 71,07** |
| CLIP thị giác gốc, một lượt cho sketch | 30,29 / 41,63 | 36,34 / 33,85 |
| Khung ảnh không in chữ | 22,15 / 34,28 | 31,46 / 27,24 |
| Cả khung ảnh và chữ, trước phép trừ | 24,15 / 34,91 | 34,41 / 28,15 |
| Chỉ tên do ảnh đề cử, đọc bằng text encoder | 41,98 / 52,36 | 54,51 / 48,50 |
| Chỉ ảnh hưởng của chữ in trên ô trống | 16,55 / 18,60 | 23,66 / 14,49 |
| **Phần suy giảm khi in cùng chữ cạnh ảnh** | **13,44 / 19,85** | **16,95 / 15,49** |
| Cùng phép đo, tráo chữ giữa các ảnh | 8,64 / 15,31 | 15,27 / 12,00 |
| Giữ nguyên chữ của từng ảnh, đổi vector đã tính giữa các ảnh có cùng chữ | 11,76 / 17,15 | 15,28 / 14,25 |

Ảnh tự đề cử 416 tên khác nhau trên Ext-1 và 339 trên Ext-2. Khi tráo toàn bộ tên, 1,71% và 2,07% ảnh tình cờ giữ đúng tên cũ. Trong đối chứng đổi vector giữa những ảnh cùng tên, chỉ 59 và 55 ảnh không có ảnh thứ hai để đổi. Giữ đúng cả ảnh và tên cho mAP cao hơn đổi vector nội nhóm **1,68 / 1,67** điểm theo trọng số truy vấn. Khoảng 95% theo lớp của hiệu ứng là **[1,00 đến 2,46]** và **[−1,76 đến 6,19]**. Như vậy chỉ mục giữ một phần thông tin riêng của ảnh, với bằng chứng rõ hơn ở Ext-1. [Điểm từng nhánh Ext-1](official-sketchy_1.json) · [Điểm từng nhánh Ext-2](official-sketchy_2.json) · [Đổi vector Ext-1](reassigned-sketchy_1.json) · [Đổi vector Ext-2](reassigned-sketchy_2.json)

## Khi ảnh đúng và sai mang cùng một chữ

Với mỗi sketch, chọn một ảnh đúng lớp khác ảnh nguồn. Trong các ảnh sai lớp mang đúng cùng danh từ, tìm một ảnh ngay phía trên và một ảnh ngay phía dưới theo cosine CLIP gốc. Trung bình hai so sánh của từng sketch trước khi cân bằng vật thể và lớp. Cosine gốc phải đạt 50% cho từng sketch. Chỉ đọc chữ cũng cho 50% vì hai ứng viên có cùng chữ. Các nhãn chỉ được dùng để lập đối chứng này sau khi có vector.

| Độ chính xác ghép cặp (%), cân bằng theo lớp và vật thể | Ext-1 | Ext-2 |
|---|---:|---:|
| Sketch đủ điều kiện / tổng | 7.301 / 14.179 | 7.585 / 12.694 |
| CLIP thị giác gốc | 50,00 | 50,00 |
| Chỉ chữ | 50,00 | 50,00 |
| Cả khung ảnh và chữ | 56,57 [53,07 đến 60,16] | 58,35 [52,25 đến 64,31] |
| **Phần suy giảm được chọn làm chỉ mục** | **50,11 [45,34 đến 55,00]** | **44,73 [38,52 đến 50,88]** |

Lấy nguyên khung có thể phân biệt cặp ảnh này, nhưng mAP toàn gallery của chính đối chứng ấy vẫn thấp hơn cosine CLIP gốc. Kết quả trên một nhóm khó không đủ để dự đoán lợi ích truy hồi toàn tập. [Cặp và kết quả từng sketch](photo-name-L4.zip)

## Vì sao giả thuyết suy giảm chỉ đúng một phần?

Gọi r là mức thay đổi vector do in chữ trên nền trắng, và g là vector chỉ mục của ảnh. Tỉ lệ ⟨g,r⟩/‖r‖² đo phần phản ứng theo hướng chữ bị mất khi có hình. Trên tất cả ảnh, chữ do ảnh tự đề cử mất trung bình **64,30%** phản ứng ở cả hai split. Chữ tráo cũng mất **60,46% / 60,35%**. Cân bằng theo lớp, phần chênh giữa chữ tự đề cử và chữ tráo là **3,85 [2,42 đến 5,51]** và **4,05 [2,29 đến 5,91]** điểm phần trăm. Đây là phép giải thích bổ sung sau khi thấy điểm chính thức. Hai con số khoảng 4 điểm này đo phản ứng nội tại của encoder.

Trung vị phần năng lượng của vector chỉ mục nằm dọc theo hướng chữ in đơn lẻ là **61,32% / 60,58%**. Dấu hiệu ảnh làm suy giảm chữ phù hợp thực sự tồn tại với phép đo này. Phần suy giảm cũng xuất hiện mạnh khi in chữ bị tráo, còn vector còn lại khớp kém với sketch. Kết quả không đạt điều kiện để chọn cơ chế này làm contribution truy hồi. [Phân tích từng ảnh Ext-1](attenuation-sketchy_1.json) · [Ext-2](attenuation-sketchy_2.json)

## Tái lập

Mã tạo chỉ mục ở `737856911291eb876b820733bde426bdb5845946`, mã chấm KD ở `74f46601bee61a55ce39adf68ac7b01ed319a326`, phép đổi vector cùng chữ ở `409eac033509b5285381ae3f8e525b4fcec300d4`, phép đo suy giảm ở `1e01b3ce8d5447b69eafd7b882b4e1bdbf538c84`. Gói chính chứa đủ ID, AP và precision từng truy vấn của mọi nhánh, tên thật và tên đã tráo của từng ảnh, cùng các cặp ảnh đối chứng. Mỗi gói có danh sách SHA-256 cho từng tệp bên trong.

| Gói kết quả | SHA-256 |
|---|---|
| [Truy hồi và đối chứng cùng chữ](photo-name-L4.zip) | `42743c6fbabe44d7f2db03cf6dd760ec0e614d324779a8264bef0bea832dccbe` |
| [Đổi vector giữa các ảnh mang cùng chữ](same-name-reassignment-L4.zip) | `40e807f1e11ce294a578dd988db23a5f59ea1b8a4f29862c53956ba5b074d52a` |
| [Phép đo suy giảm trên từng ảnh](attenuation-L4.zip) | `8295be8c5cb72ade2826547e5a30ed81bdb0eb26826b158c18847a2454296738` |
