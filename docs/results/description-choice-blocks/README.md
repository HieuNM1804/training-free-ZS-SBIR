# Khi nào CLIP có đủ bằng chứng để chọn một mô tả giúp tìm ảnh?

**Chọn hoàn hảo nghĩa của hai danh từ mà sketch đề cử vẫn chỉ đạt 57,51 / 63,00 mAP**, so với 75,35 / 72,66 của Semantic Dot. Khi nhãn thật được dùng để chọn câu tốt nhất, cần mở tới 40 danh từ ở Ext-1 và 6 ở Ext-2 mới có thể vượt baseline 4 điểm với cách xếp danh từ này. Cùng quy tắc trên chính ảnh nguồn chỉ cần 5 và 2. Đây là bằng chứng về khác biệt khi đề cử giữa hai miền.

Bốn block cuối tăng mạnh khả năng đề cử một mô tả hữu ích khi đọc bằng bộ mã hóa văn bản của CLIP. Diễn biến điểm qua các block còn mang một tín hiệu phân biệt lớp nhỏ khi cosine cuối được cân bằng. Phép thử đối chứng xáo thứ tự trên Ext-1 có khoảng tin cậy chứa 0. Kết quả hiện tại chưa đủ để lấy diễn biến ấy làm phương pháp truy hồi độc lập.

## Dữ liệu và thước đo

OpenAI CLIP ViT-B/32 đóng băng, suy luận trên Kaggle với bốn NVIDIA L4, seed 42. Ext-1 có 14.179 sketch, 2.500 ảnh nguồn, 17.100 ảnh gallery và 25 lớp. Ext-2 có 12.694 sketch, 2.100 ảnh nguồn, 12.552 ảnh gallery và 21 lớp. Con số của mỗi split bao gồm mọi truy vấn, trừ phép so có điều kiện được ghi riêng. Hai split dùng chung một phần dữ liệu Sketchy. TU-Berlin được dành cho một phương pháp đã khóa.

Các bảng truy hồi ghi mAP / precision (%). Ext-1 dùng mAP@all / P@100. Ext-2 dùng mAP@200 / P@200. Mỗi định nghĩa được chấm trên toàn gallery bằng evaluator KD-SBIR gốc. Việc tra lại đúng câu và lớp cho ra AP và precision của truy vấn đó. Kho từ đã khóa có 1.135 nghĩa WordNet ứng với 512 tên công khai thuộc Open Images. Khi giải thích phép đo, “ảnh nguồn” là ảnh thật mà người vẽ dùng. Ảnh này, nhãn lớp và sketch khác chỉ xuất hiện trong các đối chứng hồi cứu. [Thiết kế](../../description-choice-blocks.md) · [Pilot WordNet và Semantic Dot](../wordnet-description-renaming/README.md)

## CLIP đề cử mô tả nào khi lần lượt đọc các block?

Cho CLIP xem một sketch. Ở mỗi độ sâu, dùng cùng phép chiếu và chuẩn hóa của mô hình để tìm định nghĩa có cosine lớn nhất với sketch trong toàn kho. Đưa định nghĩa ấy qua prompt ảnh chụp để tìm trực tiếp ảnh RGB. Bảng ghi kết quả khi chỉ gửi một định nghĩa. Các mốc được chọn trước khi đo.

| Cách truy vấn ảnh RGB | Ext-1 | Ext-2 |
|---|---:|---:|
| Semantic Dot, mốc độc lập | **75,35 / 77,10** | **72,66 / 71,07** |
| Sketch, đầu ra block 4 | 7,11 / 4,01 | 5,21 / 4,53 |
| Sketch, đầu ra block 8 | 5,39 / 4,11 | 5,16 / 4,52 |
| Sketch, đầu ra cuối chuẩn | 48,52 / 55,91 | 48,71 / 45,59 |
| Sketch gốc và lật, đầu ra cuối | 49,18 / 56,61 | 49,30 / 46,16 |
| Ảnh nguồn, cùng prompt drawing khi đề cử | 64,85 / 77,14 | 71,05 / 67,76 |

Từ block 8 tới đầu ra cuối, mức tăng khi cân bằng theo vật thể rồi theo lớp là **43,02 [34,83 đến 51,28]** và **43,53 [32,61 đến 53,41]** điểm AP. Các khoảng 95% dùng bootstrap theo lớp với 10.000 lượt lấy mẫu. Kết quả xác định thời điểm cách đọc bằng ngôn ngữ này cho thông tin có ích, và để ngỏ lượng thông tin thị giác có thể đọc bằng một cơ chế khác. Mỗi giá trị được lấy từ một lượt forward bình thường. Việc chấm một biểu diễn trung gian không đo tác động nhân quả khi xóa một block. [Ext-1](summary-sketchy_1.json) · [Ext-2](summary-sketchy_2.json) · [Phân rã đóng góp trong CLIP, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5085a2b8c298edeadc46b9ffe6df5f64-Abstract-Conference.html)

## Nếu cho sẵn hai định nghĩa, đổi người chọn làm thay đổi gì?

Giữ nguyên hai định nghĩa do một sketch của **vật thể khác cùng lớp** đề cử. Mỗi đầu vào ở bảng sau chọn giữa chính hai phương án ấy. AP/P đo việc dùng mô tả được chọn để truy vấn gallery. Đối chứng cho sẵn lớp giúp tách bước đề cử khỏi bước lựa chọn. Một hệ thống triển khai phải tự có được câu hỏi ấy.

| Đầu vào quyết định giữa hai định nghĩa | Ext-1 | Ext-2 |
|---|---:|---:|
| Sketch hiện tại | 48,59 / 56,44 | 50,35 / 47,25 |
| Ảnh nguồn thực | 51,34 / 59,74 | 55,19 / 52,17 |
| Ảnh khác vật thể, cùng lớp | 51,31 / 59,75 | 55,30 / 52,20 |
| Ảnh khác lớp | 39,02 / 46,09 | 44,75 / 41,55 |

Trên chính các câu hỏi này, đổi ảnh nguồn thành ảnh khác cùng lớp chỉ làm thay đổi mAP cân bằng khoảng **0,08** và **−0,07** điểm khi tính theo chiều ảnh nguồn trừ ảnh cùng lớp. Cả hai khoảng 95% chứa 0. Khi các đầu vào được tự đề cử trong toàn kho, khoảng cách sketch–ảnh nguồn lớn hơn nhiều. Nhãn thật được dùng để chọn ảnh đối chứng, nên kết quả chưa cho phép dùng một ảnh lân cận bất kỳ thay ảnh nguồn. [Số liệu Ext-1](summary-sketchy_1.json) · [Số liệu Ext-2](summary-sketchy_2.json)

## Diễn biến qua 12 block phân biệt được ảnh nào?

Mỗi sketch được ghép với một ảnh đúng lớp khác vật thể và một ảnh sai lớp. Hai ảnh được ghép gần nhau theo ba đại lượng ở đầu ra cuối, gồm cosine với sketch, mức nghiêng về một trong hai định nghĩa, và giá trị theo một hướng ngẫu nhiên. Dọc theo CLIP, ta đo sự thay đổi của mức nghiêng về định nghĩa ở mỗi block. Trước khi so sánh, trừ diễn biến trung bình theo block của ảnh và bỏ trung bình theo độ sâu của từng hình. Chọn ứng viên có diễn biến giống sketch hơn. Thử đối chứng bằng cách sắp xếp các giá trị hoặc xáo block của mỗi ảnh. Đây là độ chính xác chọn ảnh trên cặp, đơn vị %, cân bằng theo vật thể và lớp. [Thiết kế và đối chứng](../../description-choice-endpoint-match.md)

| Phép chọn giữa ảnh đúng lớp và ảnh sai lớp | Ext-1 | Ext-2 |
|---|---:|---:|
| Cosine cuối trên cặp ghép gần nhất | 67,20 | 61,70 |
| Diễn biến theo định nghĩa trên chính cặp đó | 53,21 | 53,59 |
| Cosine cuối khi bắt buộc thắng đúng một trong hai cặp của mỗi sketch | 50,00 | 50,00 |
| Diễn biến theo định nghĩa, đối chứng cân bằng cosine | **52,72** | **53,03** |
| Sắp xếp 12 giá trị, cùng đối chứng | 48,73 | 46,93 |
| Xáo vị trí block độc lập theo ảnh, cùng đối chứng | 50,29 | 50,40 |
| Đọc theo một hướng ngẫu nhiên, cùng đối chứng | 50,51 | 50,18 |

Đối chứng cân bằng giữ một ảnh đúng lớp và tìm hai ảnh sai. Cosine gốc phải xếp một ảnh sai dưới ảnh đúng và ảnh kia trên ảnh đúng. Mỗi vế có trọng số một nửa cho từng sketch. Có **12.296/14.179** và **11.845/12.694** sketch đủ điều kiện, bao phủ đủ 25 và 21 lớp. Các sketch thiếu một vế được liệt kê trong dữ liệu. Việc cân bằng dấu của cosine không đồng nhất toàn bộ vector đầu ra hay khoảng cách giữa hai điểm.

So với 50%, diễn biến theo định nghĩa tăng **2,72 [0,40 đến 5,00]** và **3,03 [1,06 đến 5,07]** điểm phần trăm. So với đối chứng xáo block, mức chênh chỉ còn **2,43 [−0,07 đến 4,93]** và **2,63 [0,42 đến 4,96]**. Trên riêng các cặp cosine xếp sai, độ chính xác là **52,57 [49,87 đến 55,18]** và **52,80 [50,90 đến 54,69]**. Trên một phép kiểm tra khác yêu cầu tìm đúng ảnh nguồn giữa hai ảnh cùng lớp, kết quả ban đầu là **49,77%** và **50,47%**. Các số này chưa chứng minh thứ tự thời gian có ý nghĩa vì hoán vị các block giống nhau cho cả hai hình vẫn bảo toàn tích vô hướng của diễn biến. [Cặp ban đầu Ext-1](profile-sketchy_1.json) · [Cặp ban đầu Ext-2](profile-sketchy_2.json) · [Đối chứng cân bằng Ext-1](bracket-sketchy_1.json) · [Đối chứng cân bằng Ext-2](bracket-sketchy_2.json)

## Chọn đúng nghĩa trong hai danh từ có đủ không?

Với mỗi sketch, giữ nguyên hai danh từ đứng đầu rồi cho biết lớp thật để chọn định nghĩa có AP cao nhất trong **tất cả các nghĩa của cả hai danh từ**. Nếu định nghĩa thuộc nhiều từ đồng nghĩa trong WordNet, từ đã đề cử được quyền dùng nó. Kho 1.135 nghĩa có 512 tên Open Images phù hợp, 508 tên từng đứng đầu ít nhất một mục từ. Bốn tên còn lại chỉ xuất hiện dưới dạng tên đồng nghĩa. Mọi giá trị trần cho AP và precision được lấy riêng cho từng metric. Đối với ảnh nguồn và sketch, dùng cùng prompt drawing trong bước đề cử. [Thiết kế](../../description-choice-head-ceiling.md)

| Quyền chọn câu để truy vấn trực tiếp gallery RGB | Ext-1 | Ext-2 |
|---|---:|---:|
| Sketch gốc và lật, một định nghĩa đứng đầu | 49,18 / 56,61 | 49,30 / 46,16 |
| Sketch gốc và lật, tốt nhất trong đúng hai nghĩa được đề cử | 55,83 / 65,25 | 59,19 / 55,62 |
| Sketch gốc và lật, tốt nhất trong mọi nghĩa của hai danh từ đó | **57,51 / 67,79** | **63,00 / 58,71** |
| Ảnh nguồn, tốt nhất trong mọi nghĩa của hai danh từ nó đề cử | 72,72 / 85,77 | 81,84 / 78,20 |
| Tốt nhất trong mọi nghĩa của toàn kho | 86,18 / 98,19 | 94,92 / 92,06 |

Riêng một lượt đọc sketch, trần trong hai danh từ là 56,88 / 67,09 và 62,63 / 58,35. So với chọn tốt nhất trong chính hai nghĩa ban đầu, việc giải quyết đa nghĩa hoàn hảo mang lại 1,72 và 3,86 điểm mAP. Ở hai lượt xem, mức tăng tương ứng là 1,69 và 3,81. Những trần này phụ thuộc cấu trúc kho từ, prompt và cách đề cử đã khóa. [Ext-1](head-ceiling-sketchy_1.json) · [Ext-2](head-ceiling-sketchy_2.json)

Số danh từ tối thiểu bên dưới là vị trí đầu tiên trên đường cong trần mà **mAP trung bình toàn split** đạt mốc. Phép tính cho biết ngân sách tối thiểu của *bộ chọn hoàn hảo* trong thứ tự đó. Với thứ tự ngẫu nhiên, dùng kỳ vọng giải tích khi rút danh từ không hoàn lại. Hai split khác tập lớp, gallery và định nghĩa mAP, nên chênh lệch 40 so với 6 không thể quy riêng cho độ khó của sketch.

| Ai xếp thứ tự danh từ? | Ext-1 đạt baseline / +4 / +5 | Ext-2 đạt baseline / +4 / +5 |
|---|---:|---:|
| Sketch gốc và lật | 18 / **40** / 50 | 5 / **6** / 7 |
| Sketch, một lượt | 19 / 42 / 52 | 5 / 7 / 7 |
| Ảnh nguồn, một lượt | 3 / **5** / 7 | 1 / **2** / 2 |
| Thứ tự ngẫu nhiên, kỳ vọng | 226 / 300 / 323 | 72 / 91 / 97 |

## CLIP làm gì khi đã có hai từ phân biệt rõ hai lớp?

Mỗi vật thể chỉ đóng góp một ảnh nguồn và một sketch được lấy bằng seed 42. Ghép hai vật thể khác lớp, mỗi vật thể xuất hiện đúng một lần. Mỗi ảnh nguồn tự đề cử một danh từ công khai. Dùng bảng AP của tên đơn trên toàn gallery để xác định những cặp mà mỗi lớp thực sự phù hợp với một tên khác. Sau đó yêu cầu gán hai từ đó cho hai hình sao cho đúng cả cặp. Việc ghép theo lớp và chọn bằng AP chỉ phục vụ chẩn đoán này.

Ext-1 có 1.029/1.250 cặp có cách gán phân biệt và bao phủ 25 lớp. Cosine sketch–tên thông thường đúng 989/1.029 (**96,11%**), cosine ảnh nguồn–tên đúng 1.014/1.029 (**98,54%**). Ext-2 có 883/1.050 cặp, đủ 21 lớp. Hai kết quả là 863/883 (**97,73%**) và 881/883 (**99,77%**). Nhiệm vụ này cho trước hai nguồn ảnh, hai từ và hai lớp phân biệt tốt. Khi truy hồi thực, một sketch phải tự đưa ảnh đúng lên giữa hàng nghìn ứng viên. Độ chính xác cao trên phép gán này cho thấy một pilot chỉ dùng hai từ xa nghĩa dễ đánh giá quá cao khả năng sửa lỗi đề cử. [Ext-1](reciprocal-sketchy_1.json) · [Ext-2](reciprocal-sketchy_2.json)

## Tái lập

Mã trích xuất ở `d5a5347160914935c0e416d597cdd879815e2497`, phép chấm theo block ở `d19cfc1a20b2f67320c906202f25808254bc765e`, ghép cặp ở `e434ea55323864824c0bcc8bffeea30715c348d9`, đối chứng cân bằng ở `b889688ff034773c28b0ea9a86135104ff461672`. Nạp lại trên GPU tái lập cả hai lựa chọn cuối của 99,59% và 99,41% sketch. Chênh cosine tối đa tới lựa chọn cũ là 0,000156. Các đường cong hai lượt xem và báo cáo cuối giữ đúng lựa chọn đã lưu ở pilot. Tệp gốc có ID và kết quả theo từng truy vấn, bảng AP/P, mã băm các shard biểu diễn và SHA-256 từng thành viên.

Phép đo trần theo danh từ dùng mã `fd6687e9dd7353ac6710d16f818445d7b8efa84c`. Một lần kiểm tra trước khi chấm phát hiện chênh lệch 512 tên / 508 tên đại diện do tên đồng nghĩa. Phiên Kaggle đầu kết thúc sau khi đã trả về số liệu và trước khi xuất gói theo truy vấn. Phiên Kaggle mới chạy lại với bốn L4. Hai phiên cho kết quả giống nhau ở mọi mốc ngân sách, bốn số đo chính của từng đầu vào, mức tăng do chọn nghĩa và các điểm trên đường cong được chép ra từ phiên đầu. Gói ZIP chứa kết quả đầy đủ của lượt tái lập. Mã tạo cặp hai từ ở `4777a3a9ec1c29d3c7d96e781fa061c4db73e9ed`. [Bản ghi từ phiên đầu](noun-first-session.json)

| Tệp kết quả | SHA-256 |
|---|---|
| [Các block và kết quả truy hồi](block-choices-L4.zip) | `9c49a2d8bc677977433525e50cea9492e435ca7a153ca98c47f4b853005dd251` |
| [Đối chứng ghép theo đầu ra cuối](endpoint-match-L4.zip) | `2954a2ead366fa89d4668b449eebde41e07d78065658f7c64e8439b52f2b2d1d` |
| [Đối chứng cosine cân bằng](cosine-bracket-L4.zip) | `7b0ef6de7e58b651d87d59124dff875fc353030b2c1577353ad4711950ac7749` |
| [Trần theo danh từ và toàn đường cong](noun-ceilings-L4.zip) | `20df0f8155dd53ca9d8754b2881d84c5ed816d47bbedd49cbe00984db8e17036` |
| [Gán hai từ cho hai vật thể](reciprocal-choices-L4.zip) | `cc18e79a905f75c53feca742945239f7ed9bb6f1a32a72d7ad24db61f90bb246` |
