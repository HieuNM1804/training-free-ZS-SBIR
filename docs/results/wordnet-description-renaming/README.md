# Đổi chéo tên trong định nghĩa WordNet để tìm ảnh từ một sketch

**Quyết định dừng pilot.** Trên toàn bộ hai split Sketchy unseen, phương pháp đạt **47,63 / 49,46 mAP**. Đối chứng dùng cùng hai định nghĩa và bỏ tên đạt **47,99 / 49,69**. Phần đổi chéo tên làm giảm mAP ở cả hai split. Phương pháp cách Semantic Dot lần lượt 27,72 và 23,20 điểm mAP.

## Cách chạy

Xuất phát từ 601 tên công khai Open Images. Quy tắc khớp từ đã khóa tìm được 512 tên trong WordNet 3.0, với 1.135 nghĩa danh từ và 1.123 câu mô tả khác nhau sau khi ẩn tên. 89 tên không có kết quả khớp chính xác sau bước chuẩn hóa, trong đó có một số tên ghép và tên có chú thích trong ngoặc. Tất cả nghĩa phù hợp được giữ. Không có bước chọn nghĩa bằng tên lớp kiểm thử.

Với từng sketch, CLIP đọc các định nghĩa đã ẩn tên rồi đề cử hai nghĩa có tên công khai khác nhau. Mỗi định nghĩa được thử với cả tên của nó và tên của phương án kia. Chỉ xác nhận lựa chọn nếu CLIP chọn cùng một nội dung ở cả hai lần. Khi hai lần trái ý, dùng lựa chọn từ định nghĩa ẩn tên để tính hiệu quả trên toàn tập. Câu định nghĩa được chọn, với mẫu ảnh chụp, trực tiếp truy vấn embedding RGB của mọi ảnh. Các nhãn và ảnh nguồn chỉ tham gia các phép đánh giá sau khi có điểm.

Kaggle dùng bốn NVIDIA L4. OpenAI CLIP ViT-B/32 đóng băng, seed 42, hai lượt xem cùng một sketch, một lượt RGB cho mỗi ảnh. Ext-1 gồm 14.179 sketch và 17.100 ảnh, Ext-2 gồm 12.694 sketch và 12.552 ảnh. Không dò siêu tham số. TU-Berlin ngoài phạm vi pilot đã duyệt. [Giao thức khóa trước khi chạy](../../wordnet-renaming-pilot.md) · [Nguồn từ điển](https://wordnet.princeton.edu/)

## Kết quả truy hồi

Mỗi ô chứa mAP / precision (%). Ext-1 dùng mAP@all / P@100. Ext-2 dùng mAP@200 / P@200. Hàm chấm là evaluator KD-SBIR gốc.

| Cách chọn nội dung để truy vấn RGB | Sketchy Ext-1 | Sketchy Ext-2 |
|---|---:|---:|
| Semantic Dot hiện tại, 601 từ | **75,35 / 77,10** | **72,66 / 71,07** |
| Chọn tên tốt nhất trong 601 tên, dùng tên tìm RGB | 57,61 / 64,54 | 60,05 / 57,20 |
| Giữa hai nghĩa đã đề cử, dùng tên đơn để chọn định nghĩa | 48,71 / 56,00 | 51,28 / 48,31 |
| Giữa cùng hai nghĩa, dùng định nghĩa ẩn tên | 47,99 / 55,64 | 49,69 / 46,70 |
| Gắn tên gốc với định nghĩa, quyết định sau một lần đọc | 48,23 / 55,65 | 50,47 / 47,43 |
| **Đổi chéo tên, giữ lựa chọn ẩn tên khi hai lần đọc bất đồng** | **47,63 / 55,16** | **49,46 / 46,41** |

Hai lần gắn tên đồng ý trên 3.841/14.179 sketch của Ext-1 (27,09%) và 3.500/12.694 của Ext-2 (27,57%). So với phần đọc định nghĩa, nó thực sự đổi phương án ở 715 và 773 truy vấn. AP tăng/giảm ở 292/423 và 313/421 truy vấn. Những lần thay mô tả còn lại trên Ext-2 hòa AP@200. Khoảng bootstrap 95% theo lớp cho hiệu ứng mAP so với định nghĩa ẩn tên là [−0,64; −0,10] trên Ext-1 và [−0,59; +0,17] trên Ext-2. Ước lượng chính dùng trọng số từng truy vấn như evaluator gốc.

Trên nhóm lỗi đã chỉ ra từ baseline, sketch và ảnh sai cùng có một từ thắng trong khi ảnh đúng tốt nhất có từ thắng khác. So đúng một cặp ảnh sai/đúng cho mỗi truy vấn, đổi tên sửa được **814/1.955** thứ tự trên Ext-1 và **747/1.898** trên Ext-2. Chỉ đọc định nghĩa đã sửa được **803** và **743**. Trong toàn bộ các cặp ở Ext-1, đổi tên sửa thêm 28 trường hợp nhưng cũng làm sai thêm 51 so với đối chứng định nghĩa. Ext-2 sửa thêm 3 và làm sai ít hơn 28. Mức đổi thứ tự cặp không chuyển thành tiến bộ mAP trên toàn gallery. [Số Ext-1](summary-sketchy_1.json) · [Số Ext-2](summary-sketchy_2.json)

## Cơ hội bị mất ở bước nào?

Sau khi chạy xong, dùng nhãn thật để hỏi mức tối đa có thể đạt nếu chọn một câu dùng cho RGB. Vì mọi sketch của một lớp có cùng tập ảnh liên quan, có thể chấm mỗi câu với evaluator gốc rồi lấy câu có AP tốt nhất cho từng lớp. Với hai ứng viên của một sketch, lấy tốt hơn trong đúng hai ứng viên ấy. Các giới hạn cho AP và precision được tính riêng. Những con số sau sử dụng nhãn thật và chỉ cho biết dư địa.

| Tập phương án mà người biết lớp thật được phép chọn | Ext-1 mAP / P@100 | Ext-2 mAP@200 / P@200 |
|---|---:|---:|
| Hai mô tả thực mà sketch đã đề cử | 54,48 / 64,19 | 59,25 / 55,76 |
| Toàn bộ định nghĩa ẩn tên trong kho gốc | 86,18 / 98,19 | 94,92 / 92,06 |
| Thêm tất cả cách ẩn tên từng cặp đã thực sự xuất hiện ở split | 86,39 / 98,23 | 95,99 / 93,17 |
| Toàn bộ 601 tên công khai, mỗi tên cho đúng một truy vấn RGB | 88,28 / 95,88 | 98,24 / 96,50 |

Ngay cả chọn hoàn hảo giữa hai mô tả đã đề cử vẫn thấp hơn Semantic Dot **20,86 / 13,42 điểm mAP**. Cả kho lại có câu tìm được ảnh rất tốt. Một bộ chọn giai đoạn cuối không thể bù cho việc ứng viên hữu ích đã bị bỏ ở bước đề cử. Trần toàn kho được tính trên chính gallery dùng chọn câu tốt nhất theo nhãn, do đó chưa chứng minh chọn được câu ấy cho lớp mới trong thực tế. [Định nghĩa phép đo](../../wordnet-gloss-ceiling.md) · [Trần Ext-1](ceiling/summary-sketchy_1.json) · [Trần Ext-2](ceiling/summary-sketchy_2.json)

## Cùng một vật thể, sketch và ảnh nguồn đề cử câu khác nhau

So từng sketch với đúng ảnh người vẽ đã nhìn, trên 2.500 ảnh nguồn của Ext-1 và 2.100 ảnh của Ext-2. Giữ một lượt nhìn ở cả hai phía và cùng prompt `drawing` khi đề cử hai mô tả. Đo AP của ứng viên tốt hơn bằng cùng kho định nghĩa ẩn tên chuẩn và cùng gallery RGB. Bảng này dùng AP của hai nghĩa chuẩn trước bước ẩn tên chéo, nên khác hàng oracle hai mô tả trong bảng trên.

| Đầu vào đề cử hai mô tả | Ext-1, AP của ứng viên tốt hơn | Ext-2, AP của ứng viên tốt hơn |
|---|---:|---:|
| Bản vẽ, prompt drawing | 55,15 | 58,77 |
| Chính ảnh nguồn, cùng prompt drawing | 72,03 | 80,15 |
| Bản vẽ, prompt photo | 53,30 | 57,44 |
| Chính ảnh nguồn, cùng prompt photo | 74,52 | 81,89 |

Với prompt drawing giữ nguyên, chênh lệch ảnh nguồn trừ sketch là **16,87 / 21,37 điểm AP** theo trọng số truy vấn. Khi cân bằng vật thể rồi cân bằng lớp, chênh lệch là **16,82 / 20,52**, với khoảng bootstrap 95% [11,19; 22,57] và [13,98; 27,39]. Thay prompt giữ nguyên chiều hiệu ứng. Đây là kiểm tra có ghép ảnh nguồn sau thực nghiệm, và chính ảnh nguồn cũng thuộc gallery. Nó cho thấy thay đổi kiểu thể hiện đi kèm sự khác biệt lớn khi CLIP đề cử mô tả có ích. Nét vẽ có thể đã bỏ chi tiết, hoặc encoder có thể không đọc tốt chi tiết còn giữ lại. Phép đo này chưa tách hai nguyên nhân. [Thiết kế ghép cặp](../../wordnet-matched-depictions.md) · [Ext-1](matched/sketchy_1.json) · [Ext-2](matched/sketchy_2.json)

Một ví dụ là 673 bản vẽ hươu cao cổ trên Ext-2. Với cùng prompt drawing, định nghĩa ẩn tên của nghĩa WordNet *giraffe* đứng đầu cho **87/673** bản vẽ và **535/673** lượt đọc ảnh nguồn tương ứng. Số liệu phía ảnh có lặp theo số bản vẽ của mỗi ảnh. Với bản vẽ, lựa chọn phổ biến nhất là *alpaca* (258/673). Chính định nghĩa *giraffe* khi gửi đi tìm ảnh lại đạt **99,56 AP@200** ở lớp này. Lựa chọn được một mô tả đúng cho ảnh chụp không bảo đảm CLIP cũng chọn được mô tả đó cho bản vẽ.

## Trên hình thực tế còn nhìn thấy gì?

Tôi xem 24 bản vẽ thuộc 12 lớp. Mỗi hàng có hai ảnh theo thứ tự ngẫu nhiên và chỉ một ảnh đúng lớp. Ảnh đúng là một vật thể khác ảnh nguồn. Tôi ghi lựa chọn A, B hoặc chưa rõ cùng dấu hiệu nhìn thấy rồi mới mở khóa nhãn. Tôi chọn đúng 19, sai 1 và để 4 chưa rõ. Trong riêng 12 trường hợp cả hai định nghĩa của pilot đều ưu tiên ảnh sai, tôi chọn đúng 10, sai 1 và để 1 chưa rõ. Đây là đánh giá định tính của một trợ lý AI, trên mẫu cố ý cân bằng lỗi và đối chứng, với cả hai ảnh cho sẵn.

Ở trường hợp 14, hai định nghĩa đều chọn ảnh bò, trong khi bản vẽ thể hiện sừng trên mũi khớp với ảnh tê giác. Ở trường hợp 16, cả hai cùng ưu tiên ảnh có ô, trong khi bản vẽ ốc sên có cả vỏ trên lưng và hai râu ở đầu. Trường hợp 18 cho thấy giới hạn của cách đọc này. Tôi nhìn nhầm hình chim hải âu thành đuôi cá heo. Một số ảnh cũng chứa nhiều vật thể, và trong trường hợp 5 ảnh còn có chữ chỉ tên cái kéo. Các ví dụ cho phép đặt giả thuyết về bằng chứng còn hiện diện nhưng bị bỏ qua, chưa thể dùng để ước lượng tần suất của lỗi đó trên toàn benchmark. [Giao thức](../../wordnet-visual-inspection.md) · [Lựa chọn trước khi mở nhãn](blind-judgments.tsv) · [Đối chiếu sau khi mở nhãn](blind-review.json) · [ID ảnh để tái lập](blind-selection-key.json)

## Đánh giá hướng nghiên cứu

Phép đổi tên là đối chứng cho việc CLIP bám tên hay bám nội dung mô tả. Kết quả chưa ủng hộ việc đưa nó thành contribution truy hồi độc lập. Việc tạo hoặc dùng định nghĩa đã có prior art trong nhận dạng bằng mô tả. *FreeDom* đã chuyển ảnh thành từ truy vấn để đổi miền. *Half-Truths* đã cho thấy một chi tiết sai còn có thể làm tăng điểm của CLIP và dùng các cặp mô tả có sửa một thành phần để kiểm tra. [Visual Classification via Description, ICLR 2023](https://cv.cs.columbia.edu/sachit/classviadescr/) · [FreeDom, WACV 2025](https://cmp.felk.cvut.cz/~efthynik/freedom/freedom) · [Half-Truths, NeurIPS 2026](https://arxiv.org/abs/2602.23906)

Khoảng trống đáng kiểm tra là liệu một chi tiết thực sự **còn nhìn thấy trong cả hai cách thể hiện** có mất sức phân biệt khi CLIP đọc nó trong toàn bộ mô tả. Chẳng hạn, so khả năng CLIP nhận ra vị trí sừng khi hỏi trực tiếp và khi cùng chi tiết nằm trong mô tả trọn con vật. Cần xác nhận chi tiết nhìn thấy trên từng ảnh trước, đồng thời dùng câu đối chứng chỉ thay đúng vị trí bộ phận. Nếu câu ngắn cũng chọn sai trên sketch, giả thuyết “thông tin đúng bị chìm trong câu” yếu đi. Nếu câu ngắn chọn đúng ở cả hai miền mà câu đầy đủ lại đẩy ảnh sai lên, có cơ sở để phát triển một cơ chế truy hồi riêng. Việc đưa tên thuộc tính hoặc quy tắc riêng từng lớp vào thuật toán không đáp ứng mục tiêu. Bài về caption phân biệt ảnh và bài về correspondence cung cấp hai đối chứng khoa học cho câu hỏi này. [Discriminability Objective, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Luo_Discriminability_Objective_for_CVPR_2018_paper.pdf) · [Learning Dense Correspondences between Photos and Sketches, ICML 2023](https://proceedings.mlr.press/v202/lu23g.html)

## Tái lập

Mã pilot ở `6fa17be0cdb20611110879dec3dafc1b31f0f798`, mã kiểm tra trần ở `50904116bfa71ad76e6ae6c65f5fec5a235d713d` và mã đối chiếu ảnh nguồn ở `7dea8e9d4e91eb7361ffb4d60752ed12155dfb76`. Bốn shard chứa đầy đủ ID, hai ứng viên, biên điểm khi đổi tên, các quyết định và AP/P của từng truy vấn. Hai phân tích hồi cứu dùng lại kho này. Trên cả hai split, hai lựa chọn từ ảnh gốc+lật được tái lập 100% khi đọc lại trong phân tích ảnh nguồn.

| Gói kết quả | SHA-256 |
|---|---|
| [Pilot và kho nghĩa](wordnet-pilot-L4.zip) | `2582ba8ed8ae65ebd4ce7e1de61c419bad438e3cde6a04dfb98f12cc78bed0c8` |
| [Trần theo lớp và toàn bộ câu](ceiling/wordnet-ceiling-L4.zip) | `f32f2953c82b9b8eb695a6ff90e65ca8cc665cb9bbc3e454d1cd5e030197af7c` |
| [Ảnh nguồn và từng sketch](matched/wordnet-matched-L4.zip) | `bc1b560f845cd4329641c735876fea62d0b7f94a4a0373f6375fdea51f36870e` |

[Giấy phép WordNet](WORDNET-LICENSE.txt)
