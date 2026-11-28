# 64 ảnh lân cận làm thay đổi phần nào của thứ hạng?

64 láng giềng nâng mAP từ **75,35 lên 80,15** ở Ext-1 và **72,66 lên 75,71** ở Ext-2. Khi nhóm ảnh theo danh từ mà ảnh tự đề cử, **3,70/4,80** và **2,27/3,05** điểm tăng được quy cho thay đổi vị trí giữa các nhóm. Đổi ngẫu nhiên danh từ giữa các ảnh **trong cùng lớp thật** vẫn tái hiện được phần tăng này. Vì vậy phép chia theo từ giúp nhìn thấy hiệu ứng theo lớp, trong khi việc mỗi ảnh mang chính xác danh từ nào chưa giải thích được lợi ích 64-NN. Đây là phân tích của một bộ truy hồi đã có.

## Hai thứ hạng được tách và ghép như thế nào?

Mỗi thứ hạng có thể mô tả bằng hai thành phần. Thành phần thứ nhất là danh từ xuất hiện ở từng vị trí. Thành phần thứ hai là thứ tự ảnh bên trong từng danh từ. Ví dụ, hai vị trí mang từ “horse” lần lượt được điền bằng ảnh xếp thứ nhất và thứ hai trong nhóm “horse”. Chương trình kiểm tra rằng ghép hai thành phần của cùng một thứ hạng khôi phục chính xác mọi ID.

Lấy hai thứ hạng đã biết là Semantic Dot và kết quả dùng 64 láng giềng RGB. Đổi riêng từng thành phần, chấm AP/P, rồi tính tác động trung bình qua hai thứ tự đổi. Hai tác động cộng lại thành đúng chênh lệch theo từng sketch. Danh từ lấy từ lần ảnh tự đề cử một định nghĩa WordNet, trước khi in chữ trong [pilot ảnh–chữ](../photo-name-interaction/README.md). Danh từ này giữ nguyên ở tất cả các thứ hạng. [Giao thức và công thức](../../gallery-name-rank-decomposition.md)

## Điểm truy hồi trên toàn gallery

Mỗi ô là mAP / precision (%). Ext-1 có 14.179 sketch × 17.100 ảnh và dùng mAP@all / P@100. Ext-2 có 12.694 sketch × 12.552 ảnh và dùng mAP@200 / P@200. Dùng OpenAI CLIP ViT-B/32 đóng băng, seed 42 trên Kaggle 4×L4. Dùng bộ phát lại thứ hạng đã kiểm tra mã hàm KD-SBIR. Hai đầu mút tái hiện số công bố trước đây với độ lệch tối đa 0,007 điểm phần trăm so với số đã làm tròn. Hai split Sketchy có dữ liệu chung. Đây là phân tích thăm dò, không có bước chọn siêu tham số hay chấm TU.

| Xếp vị trí các nhóm tên | Xếp ảnh bên trong mỗi nhóm | Ext-1 | Ext-2 |
|---|---|---:|---:|
| Semantic Dot | Semantic Dot | 75,35 / 77,10 | 72,66 / 71,08 |
| 64 láng giềng | Semantic Dot | 78,79 / 79,60 | 74,94 / 72,96 |
| Semantic Dot | 64 láng giềng | 76,19 / 77,65 | 73,45 / 71,71 |
| 64 láng giềng | 64 láng giềng | **80,15 / 79,94** | **75,71 / 73,60** |

P@200 của Semantic Dot trong gói tham chiếu là 71,07, còn lần phát lại ở đây cho 71,08 sau làm tròn. Các hàng ghép chỉ giúp phân tích vì dùng thông tin từ cả hai bảng xếp hạng. [Kết quả Ext-1](summary-sketchy_1.json) · [Kết quả Ext-2](summary-sketchy_2.json)

## Danh từ giúp vì điều gì?

Bảng ghi đóng góp theo mAP, với trọng số truy vấn của evaluator. Cách chia nhóm là biến duy nhất thay đổi.

| Cách gán tên cho ảnh | Ext-1 giữa nhóm | Ext-1 trong nhóm | Ext-2 giữa nhóm | Ext-2 trong nhóm |
|---|---:|---:|---:|---:|
| Danh từ ảnh tự đề cử | **3,70** | 1,10 | **2,27** | 0,78 |
| Tráo trên toàn gallery, giữ kích thước từng nhóm tên | 0,92 | 3,88 | 0,66 | 2,38 |
| Tráo trong từng lớp thật, giữ đúng số ảnh của từng lớp ở mỗi nhóm tên | **3,86** | 0,94 | **2,61** | 0,44 |

So sánh đầu tiên cho thấy tên thật phân nhóm các ảnh liên quan. Để kiểm tra điều gì đằng sau kết quả ấy, tôi tráo tên tám lần độc lập **trong từng lớp**, kiểm tra toàn bộ bảng đếm lớp × tên trước và sau, rồi lấy trung bình tám lượt. Chỉ khoảng 34% ảnh Ext-1 và 30% ảnh Ext-2 tình cờ giữ tên cũ. Tuy nhiên, phần tăng giữa nhóm ở cả tám lượt vẫn đạt 3,81–3,91 và 2,47–2,73 điểm.

Với trọng số vật thể và lớp bằng nhau, phần đóng góp giữa nhóm của tên thật trừ đối chứng giữ thành phần lớp là **−0,16 [−0,28 đến −0,05]** và **−0,36 [−0,77 đến −0,03]** điểm mAP. Các khoảng bootstrap 95% lấy mẫu lại lớp 10.000 lần. Việc giữ thành phần lớp đã đủ để tái tạo, thậm chí hơi vượt, hiệu ứng giữa nhóm của cách gán tên thật trên hai split này. Điều này chưa chứng minh 64-NN tương đương biết nhãn, hay mọi chi tiết riêng của ảnh đều vô ích. [Giao thức đối chứng](../../gallery-name-category-control.md) · [Tám lượt Ext-1](class-control-sketchy_1.json) · [Tám lượt Ext-2](class-control-sketchy_2.json)

## Ý nghĩa đối với một phương pháp độc lập

Tín hiệu đáng theo đuổi là khả năng chọn **nhóm ảnh chứa đúng loại vật thể**, kể cả các ảnh trông khác ví dụ ban đầu. Một tên tốt trên từng ảnh, một cặp từ dễ phân biệt, hay một chữ kích hoạt CLIP mạnh đều có thể đạt kết quả thuận lợi trong phép đo riêng. Để làm cơ sở cho một cơ chế mới, bằng chứng phải dự báo được lựa chọn đúng trên **vật thể khác** và duy trì trên các ảnh cạnh tranh gần ranh giới lớp. Khả năng mô tả thứ hạng bằng nhãn thật chỉ đặt ra câu hỏi này. Một đóng góp chính phải cung cấp cách ước lượng quan hệ đó từ dữ liệu được phép dùng lúc suy luận.

Mã phân rã ở `f9450fb18afc16d4dd0c3a75624978117052002a`, mã đối chứng theo lớp ở `235f3b5f69fc7576a8e321e17a8693340d5ef6cc`. Hai gói chứa dữ liệu theo tất cả truy vấn, ID, cách phân nhóm và mã băm từng thành viên.

| Gói kết quả | SHA-256 |
|---|---|
| [Phân rã và đối chứng tráo tự do](gallery-name-order-L4.zip) | `a34ec00769a19898efdb965a0c40749d732c04c742fd416ed4c566ce0813387d` |
| [Tám đối chứng giữ thành phần lớp](gallery-name-class-control-L4.zip) | `ed1ba8765219cff6edbc434bb058de2e193ebbc113960f89325126d14821a981` |
