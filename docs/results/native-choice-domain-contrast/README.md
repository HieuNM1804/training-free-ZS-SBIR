# CLIP giữ lại thông tin gì khi một sketch chọn sai ảnh?

Chẩn đoán đặt hai câu hỏi. Nếu giữ nguyên hai ảnh, sketch của hai lớp có giúp CLIP ghép đúng chúng dù ban đầu CLIP chọn sai không? Khi sketch và ảnh xuất phát từ cùng một vật thể, việc đổi tên miền trong câu có dự báo được phần thay đổi của biểu diễn thị giác sau khi đã biết danh từ không?

## Cặp ảnh thật cạnh tranh ở hạng đầu

Chấm mọi sketch trên gallery của split. Lấy ảnh A có điểm cosine cao nhất ở lớp đúng và ảnh B có điểm cao nhất ở các lớp sai. Trên Sketchy, bỏ ảnh nguồn của chính sketch khi xác định hai ứng viên. Đây là phép đo cơ chế sau khi đánh giá, nên hạng đầu ở đây được hiểu theo điều kiện loại ảnh nguồn. Với mỗi sketch q thuộc lớp A, đo mức nó ưu tiên A hơn B. So mức này với tất cả bản vẽ của lớp B, loại bản vẽ của chính B nếu B cũng là ảnh nguồn. Mỗi ảnh nguồn có trọng số bằng nhau, các bản vẽ cùng nguồn chia đều trọng số. Đối chứng dùng sketch thuộc một lớp khác cả A lẫn B, với trọng số bằng nhau giữa các lớp. Lần lượt cân bằng theo ảnh nguồn rồi theo lớp của query trước khi tổng hợp. Phép kiểm tra dùng nhãn và các sketch khác sau xếp hạng.

| Trong các lỗi top-1 của CLIP thị giác | Ext-1 | Ext-2 |
|---|---:|---:|
| Số sketch bị xếp sai, từ toàn bộ tập | 9.004 / 14.179 | 7.983 / 12.694 |
| Tỷ lệ ghép đúng với sketch thuộc lớp B | 94,67% | 91,28% |
| Cùng phép so, dùng sketch thuộc lớp khác cả A và B | 79,64% | 74,85% |
| Chênh lệch điểm phần trăm, bootstrap 95% theo lớp | +15,03 [11,52 · 18,68] | +16,43 [12,13 · 21,15] |
| Ghép đúng khi cả lớp A và B đều riêng cho split | 82,38% | 76,29% |
| Đối chứng lớp khác trên cùng tập lớp riêng | 52,12% | 53,14% |

Bốn lớp chung là scissors, tree, wheelchair và windmill. Lọc cả hai phía của phép so giữ lại 841 và 2.257 lỗi. Sự sụt giảm cho thấy lựa chọn những lớp thực sự cạnh tranh ảnh hưởng lớn đến tỷ lệ. Mức chênh so với đối chứng vẫn dương ở cả hai nhóm lớp riêng. Các ảnh cạnh tranh vẫn được chọn trên toàn gallery trước khi lọc. Khoảng bootstrap là độ biến thiên theo lớp trong mẫu đã quan sát và không đại diện cho mọi miền ảnh. [Khoảng theo lớp](sketchy-choice-intervals.json) · [Các hàng Semantic Dot trên cùng hai ứng viên](sketchy-choice-summary.json)

## Giới hạn của phép suy luận

Gọi m(q) là điểm của A trừ điểm của B. Phép ghép đúng khi m(q_A) lớn hơn m(q_B). Với cosine trên các vector đơn vị, hiệu này bằng tích vô hướng của hai hiệu vector. Cộng một thiên lệch bất kỳ cho mỗi ảnh hoặc cho mỗi query đều được triệt tiêu. Đây cũng là cách đánh giá phân biệt thứ tự ghép trong một số nghiên cứu đa phương thức. Chỉ làm được phép triệt tiêu trên hai truy vấn đã biết chưa tạo được thuật toán truy hồi trên một sketch. [Winoground, CVPR 2022](https://arxiv.org/abs/2204.03162) · [Nghiên cứu gần đây về quyết định ghép cặp của CLIP, preprint](https://arxiv.org/abs/2611.30538)

Một phản ví dụ có ba lớp minh họa tại sao xác suất ghép đúng cao chưa chứng minh chỉ cần trừ một độ lệch cố định cho từng ảnh. Lấy ba ảnh ứng với ba vector đơn vị trực giao. Điểm của ba query trước khi chia cho cùng một hằng số căn 5 lần lượt là (0,1,−2), (−2,0,1), (1,−2,0). Mỗi cặp hai lớp ghép đúng vì tổng đường chéo bằng 0 và tổng ghép chéo bằng −1. Cả ba query vẫn chọn sai khi đứng riêng. Muốn sửa cả ba bằng độ lệch ảnh, phải có b_A−b_B, b_B−b_C và b_C−b_A cùng lớn hơn 1/√5. Tổng ba vế trái bằng 0. Tương tự, với các ảnh đơn vị, lấy trung bình (A+B)/2 làm query thay thế trong hiệu hai điểm không thay đổi quyết định vì (A+B) vuông góc với (A−B).

## Can thiệp có kiểm soát vào đầu vào của CLIP

Ở Sketchy, lấy ảnh chụp đã biết của từng vật thể và trung bình các bản vẽ của chính nó, rồi tính độ dịch chuyển thị giác theo lớp, các vật thể có trọng số bằng nhau. Một đối chứng vật lý giữ nguyên vật thể và bố cục bằng cách mã hóa ảnh RGB và ảnh xám từ cùng raster. Ở phía ngôn ngữ, giữ cùng danh từ và đổi drawing/photo hoặc black and white photo/color photo. Dùng cả hai mạo từ a/the để tách thay đổi mạo từ ra khỏi thay đổi miền. Tên lớp thật được dùng để kiểm chứng giả thuyết trong điều kiện thuận lợi nhất. Với từng lớp giữ lại, mô hình đối chứng được phép biết danh từ thông qua vector trung bình của các câu. Kiểm tra phần mà hiệu hai prompt còn dự báo được ngoài đối chứng này. Hệ số chỉ được ước tính từ các lớp khác. [Thiết kế và tiêu chí trước lần đo](../../native-choice-domain-contrast.md)

| Giảm tương đối sai số bình phương ngoài đối chứng danh từ | Ext-1 | Ext-2 |
|---|---:|---:|
| Dự báo thay đổi sketch → photo ở lớp giữ lại | −0,001% | +0,062% |
| Khoảng bootstrap 95% theo lớp | [−0,063% · +0,061%] | [−0,125% · +0,237%] |
| Dự báo thay đổi ảnh xám → RGB của cùng ảnh | +0,038% | +0,057% |
| Dự báo độ đổi biên điểm của chính hai ảnh ở các lỗi top-1 | −0,214% | −0,344% |

Hai hàng dưới là hai phép thử riêng. Hàng ảnh xám là thay đổi được kiểm soát chính xác trên mọi ảnh gallery. Hàng cuối thay sketch bằng ảnh nguồn và đo chênh điểm trên hai ảnh đã chọn từ trước. Các số là tỷ lệ giảm sai số bình phương, giá trị âm nghĩa là dự báo tệ đi. Hệ số cố định từ Ext-1 khi áp sang 17 lớp riêng của Ext-2 chỉ giảm sai số 0,078% với khoảng [−0,012% · +0,160%]. Giả thuyết cộng trực tiếp hướng thay đổi ngôn ngữ với một hệ số chung thiếu lợi ích ổn định. Phép đo này vẫn để ngỏ quan hệ phi tuyến giữa hai encoder. [Số liệu, đối chứng mạo từ và hoán vị tên lớp](sketchy-language-effects.json)

LADS học phép biến đổi đặc trưng ảnh có hướng dẫn từ mô tả miền và mục tiêu giữ lớp. Việc chỉ cộng hiệu prompt là một giả thuyết đơn giản hơn kỹ thuật của LADS. FreeDom cải thiện đảo ngược bằng từ qua các ảnh proxy, còn BASIC chấm riêng ảnh và văn bản. Kết quả của chúng đặt ra những đối chứng cho các hướng dùng từ, proxy và fusion. [LADS, ICLR 2023](https://lisadunlap.github.io/LADS-website/) · [FreeDom, WACV 2025](https://cmp.felk.cvut.cz/~efthynik/freedom/freedom) · [BASIC, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/6391a02f02dc529cfa3f242fb4343c6a-Abstract-Conference.html)

## Chuyển giao sang TU-Berlin với giả thuyết đã khóa

Dùng 2.130 sketch và toàn bộ gallery 27.262 ảnh của 30 lớp. TU-Berlin không cung cấp đối sánh sketch với ảnh nguồn như Sketchy, nên trong từng lớp mỗi sketch có trọng số bằng nhau. Trong số 835 lỗi top-1 của cosine thị giác, khi ghép hai sketch với hai ảnh, xác suất chọn đúng là **88,46%** nếu sketch đối chứng thuộc lớp của ảnh sai và **62,68%** nếu nó thuộc lớp thứ ba. Chênh lệch bằng **25,78 điểm phần trăm**, khoảng bootstrap 95% theo lớp [21,33 · 30,26]. Dấu của hiệu ứng chuyển sang TU-Berlin như dự đoán. Tỷ lệ này đòi hỏi biết lớp của sketch đối chứng và chỉ dùng để phân tích. [Kết quả gốc của TU](choices-tuberlin.json) · [Giao thức và hai dự đoán được khóa](../../native-choice-transfer-lock.md)

Khóa cả hai hệ số bằng ảnh và sketch của Ext-1, áp nguyên sang TU. Thêm hướng thay đổi của prompt ngoài đối chứng biết danh từ làm giảm sai số bình phương khi dự báo chuyển sketch → photo ở trung bình lớp **0,019%** [−0,018% · +0,054%]. Đối chứng cùng ảnh xám → RGB cho **−0,070%** [−0,169% · +0,020%]. Mức thay đổi trực tiếp mà hiệu prompt giải thích thêm rất nhỏ ở cả hai phép đo. Vì TU thiếu cặp nguồn cho từng bản vẽ, phần sketch → photo ở đây xét dịch chuyển của **hai trung bình lớp**. Kết quả không cho biết chuyện gì xảy ra với cùng một cá thể khi được vẽ lại. [Sai số đầy đủ và hệ số nguồn](language-locked-transfer.json)

## Một giả thuyết về cơ chế bên trong CLIP

Số liệu trên cho biết đầu ra còn mã hóa tương phản giữa hai vật thể. Nó chưa chỉ ra tầng nào tạo ra lỗi. Một ứng viên được **giữ ở mức đề xuất theo quyết định của người dùng** là dùng cách các patch khiến CLIP phản ứng với hai trạng thái tổng hợp khác nhau làm biểu diễn truy hồi. Ở giữa ViT-B/32 sau block 6, giữ 49 trạng thái patch của hình x. Chạy sáu block còn lại với hai token CLS lấy từ hai ảnh tham chiếu không nhãn khác nhau. Mỗi ảnh gallery và mỗi sketch sẽ được đo với cùng bốn cặp tham chiếu cố định trong split, lấy theo seed 42 trước khi xem nhãn. Có thể lập chỉ mục ảnh trước và chấm từng sketch độc lập. Ứng viên này chưa được triển khai hay chấm điểm.

Gọi F(c,P) là trạng thái CLS cuối, ngay trước LayerNorm, khi bắt đầu nửa sau mạng với CLS c và patch P. Với cặp tham chiếu a,b, phép đo trên x là F(c_a,P_x)−F(c_b,P_x). Nếu F tách được thành u(c)+v(P), mọi ảnh cho cùng kết quả. Thành phần cộng phát sinh từ việc chọn hai ảnh tham chiếu tự triệt tiêu khi so hiệu của hai hình. Vì vậy, muốn coi phép đo này là một đóng góp, tương tác còn lại phải phân biệt được *hai vật thể khác nhau cùng lớp qua hai miền*. Việc khớp đúng ảnh nguồn, nhận biết phong cách hoặc lặp lại điểm tổng hợp cũ đều chưa đủ.

Phép sàng lọc đề xuất dùng cùng các lượt đọc để so với trung bình hai đầu ra, đồng thời so với CLS gốc ở cùng tầng, đối chứng giữ độ lớn nhưng phá liên kết tham chiếu, và ảnh khác lớp. Chỉ theo tiếp nếu tương tác mang thêm tín hiệu ở cả hai split Sketchy. Khi chấm full sẽ báo cả chi phí. Một hình cần một lượt qua sáu block đầu và tám lượt qua sáu block cuối, trước tối ưu hóa theo batch. Một ví dụ về prior art là thay trạng thái giữa hai forward để xác định vai trò nhân quả. CaiT tách mô hình hóa patch khỏi việc đọc chúng bằng token tổng hợp. Trong repo cũng đã có thử nghiệm cắt các đường truyền giữa CLS và patch. Phần cần chứng minh ở đây là sự nhất quán của **đáp ứng với phép thay thế** giữa hai miền và giá trị của nó cho truy hồi độc lập. [Interchange interventions, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/4f5c422f4d49a5a807eda27434231040-Abstract.html) · [CaiT, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Touvron_Going_Deeper_With_Image_Transformers_ICCV_2021_paper.pdf) · [Đối chứng đường truyền đã chạy](https://github.com/HieuNM1804/training-free-ZS-SBIR/blob/002d841ebf219fe1dceded984e3001bb03ab3714/docs/results/semantic-context-communication/protocol.md)

Backbone và dữ liệu đều đã được khóa. Hai split Sketchy từng được dùng để thăm dò. Phép can thiệp màu tạo một thay đổi có kiểm soát từ các ảnh hiện có. Sketchy và TU-Berlin đều đánh giá chiều sketch → photo. [Khóa trước TU-Berlin](../../native-choice-transfer-lock.md)

## Tái lập

Toàn bộ phép mã hóa và đo thực nghiệm chạy trên Kaggle, bốn NVIDIA L4, OpenAI CLIP ViT-B/32 và seed 42. Bản giao thức có commit `13d022f`, mã ghép cặp `efe9cab`, mã đo ngôn ngữ `381f5af` và commit khóa trước TU là `1899abf`. Khi phiên Kaggle đầu ngắt, lượt chạy lại dùng cùng checkout đã khóa, các ID đối thủ và checkpoint đã xác minh. Trên cả hai split Sketchy, 64 giá trị tổng hợp được so với bản ghi của phiên đầu và trùng hoàn toàn. Trên TU, cosine nhỏ nhất giữa hai lần đọc đối chứng của cùng ảnh màu là 0,9999803.

Tệp ZIP gồm các đo lường trên từng query, quan sát màu, tên lớp, các vector dùng cho hồi quy và mã băm theo shard. Đã xác minh từng tệp tensor với SHA-256 trong manifest và so từng bảng JSON với bản xuất độc lập. Tệp ZIP có SHA-256 `992269eb9f654c9a1c1ad9f7c95b706f34d4de98c0537ac374e8eea49f320544`. [Dữ liệu phát lại trên Kaggle](native-choice-replay-L4.zip) · [Ext-1](choices-sketchy_1.json) · [Ext-2](choices-sketchy_2.json) · [Kiểm tra ngôn ngữ trên hai split](language-sketchy.json)
