# Dùng bố cục để kiểm tra cách CLIP diễn giải truy vấn thành từ

Cùng một cấu hình tăng mAP trên cả hai split Sketchy và TU-Berlin. Khi loại khỏi phép tính mọi sketch từng xuất hiện trong một trong hai pilot, mức tăng còn +0,20 điểm trên Ext-1 và +0,51 điểm trên Ext-2. Trên 30 lớp TU, mức tăng là +0,99 mAP và +0,97 P@100. Đây là chuyển giữa hai bộ dữ liệu cùng chiều sketch → photo.

| Hai metric (%) | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin |
|---|---:|---:|---:|
| Số sketch / số ảnh | 14.179 / 17.100 | 12.694 / 12.552 | 2.130 / 27.262 |
| Đối chứng mạnh, CLIP hai hướng và bằng chứng vùng | 85,29 / 84,65 | 77,56 / 75,46 | 80,70 / 81,49 |
| Cập nhật từ bằng bố cục đã trừ phản hồi ảnh trống | **85,48 / 84,76** | **78,14 / 76,00** | **81,69 / 82,46** |
| Xáo tọa độ patch của ảnh gallery | 85,28 / 84,54 | 77,20 / 75,19 | 80,75 / 80,66 |
| Giữ bố cục thật, bỏ hiệu chỉnh ảnh trống | 85,22 / 84,50 | 77,79 / 75,54 | 81,45 / 82,78 |
| Trừ phản hồi ảnh trống của một ứng viên khác | 85,19 / 84,56 | 77,80 / 75,60 | 80,99 / 82,11 |
| Xếp các ảnh cung cấp bằng chứng theo cosine toàn ảnh | 84,80 / 84,11 | 77,22 / 75,04 | 80,58 / 81,47 |
| Xếp các ảnh cung cấp bằng chứng theo điểm đối chứng mạnh | 85,35 / 84,32 | 77,20 / 75,35 | 81,18 / 81,53 |
| Trộn trực tiếp 10% từ do các ảnh chọn bằng bố cục cung cấp | 83,56 / 82,84 | 77,43 / 75,53 | 79,28 / 80,30 |

Ext-1 và TU dùng mAP@all / P@100. Ext-2 dùng mAP@200 / P@200. Hàm KD-SBIR gốc chấm từng thứ hạng, đối chiếu với phép tính trực tiếp. Mọi thí nghiệm chạy trên Colab Tesla T4 với cùng checkpoint OpenAI CLIP ViT-B/32. Sketchy unseen dùng để phát triển. Cấu hình được khóa ở commit `803720d` trước khi tính điểm TU cho phương pháp này. TU đã xuất hiện trong lịch sử dự án nên đây là phép kiểm tra chuyển giao cấu hình đã khóa. [Khóa và pilot](freeze.md) · [Ext-1](sketchy_1-official.json) · [Ext-2](sketchy_2-official.json) · [TU](tuberlin-official.json).

## Từ một lỗi truy hồi đến cơ chế

Khi bộ truy hồi mạnh đưa cả lớp đúng ra ngoài top 200 trên TU, cosine của CLIP vẫn tìm được ít nhất một ảnh đúng cho 188 trong 240 sketch. Trong các ảnh đúng này, trung bình theo sketch có 83,1% nằm trong nửa gallery được prompt của CLIP đánh giá giống tranh vẽ hơn. Tỷ lệ nền trong gallery của chính các lớp đó là 51,2%. Một ảnh giống sketch có thể giúp tìm nhiều ảnh chụp cùng nội dung. Chọn nhầm ảnh trung gian sẽ truyền xác suất sang lớp khác. Ở thí nghiệm một ảnh trung gian, ngay cả việc dùng nhãn để biết chính xác khi nào nên bật thay đổi điểm cố định cũng chỉ có thể tăng tối đa 0,39 mAP trên TU. [Thí nghiệm định vị lỗi](https://github.com/HieuNM1804/training-free-ZS-SBIR/blob/ca3808c1b4412a0b1ccbc12973de68d1b50c733c/docs/results/depiction-pivot-geometry/README.md).

1. Với mỗi sketch, lấy 200 ảnh gần nhất bằng cosine toàn ảnh của CLIP. Với mỗi cặp sketch–ảnh, dùng 49 patch gốc của hai ảnh và độ tương đồng CLIP để xây phân bố các cặp patch khớp. Lấy hai cặp từ phân bố này rồi đo tương quan Pearson giữa khoảng cách của hai patch bên sketch và khoảng cách tương ứng bên ảnh. Kỳ vọng được tính trực tiếp từ bảng xác suất. Lặp phép đo khi thay sketch bằng ảnh đồng nhất có cùng màu RGB trung bình, rồi lấy hiệu hai điểm. Một ứng viên có điểm cao khi hình vẽ thật cho thấy sự nhất quán bố cục cao hơn phản hồi từ lưới patch và nền đồng màu.
2. Giả sử bốc ngẫu nhiên tám ảnh có hoàn lại từ nhóm ứng viên rồi giữ ảnh có điểm cao nhất. Tính chính xác xác suất mỗi ảnh thắng, kể cả khi hai điểm bằng nhau. Với từng danh từ công khai, tính mức xuất hiện của từ đó ở các ảnh thắng so với ngay trong tập ứng viên được cosine đề cử. Cả hai đại lượng được làm trơn bằng cùng một phân bố từ trung bình. Nhờ vậy một từ phổ biến sẵn trong nhóm đề cử cần thêm bằng chứng để tăng trọng số.
3. Nhân xác suất từ ban đầu của sketch với tỷ lệ trên rồi chuẩn hóa, so với phân bố từ của ảnh đích và truyền bằng chứng qua đồ thị RGB hiện có. Gallery được chia đôi cố định. Bước tính từ cho ứng viên chỉ dùng ảnh ở nửa còn lại. Diffusion sau đó dùng toàn gallery. Điểm cuối thay phần bằng chứng từ trong đối chứng mạnh ở hệ số kế thừa 0,65. Mỗi truy vấn chỉ dùng sketch của chính nó và gallery không nhãn. Từ điển là 20.931 danh từ Open Images. Các prompt sketch/drawing và photo có mặt trong cấu hình chính. [Công thức và đối chứng](protocol.md).

## Thành phần nào tạo ra mức tăng?

Đối chứng xáo vị trí giữ nguyên nội dung patch. Xét hiệu mAP theo lớp giữa vị trí thật và vị trí xáo, khoảng bootstrap 95% là −0,05 đến +0,52 điểm ở Ext-1, +0,17 đến +2,16 ở Ext-2, và +0,21 đến +1,78 ở TU. Những khoảng tương ứng giữa phương pháp đầy đủ và hình học chưa trừ ảnh trống đều chứa 0. Bằng chứng về tác dụng của bố cục rõ nhất trên Ext-2 và TU. Bằng chứng về tác dụng riêng của phép trừ ảnh trống còn yếu. Kiểm chứng trên 2.231 sketch chung của hai split còn cho thấy chỉ 3 sketch được sửa top-1 ở cả hai gallery dù toàn bộ ảnh dương của bốn lớp chung giống hệt nhau. [Kiểm tra truy vấn chung](overlap-diagnostic.json).

| Chỉ dùng khớp từ, không thêm đối chứng vùng | Ext-1 | Ext-2 | TU |
|---|---:|---:|---:|
| Khớp từ trực tiếp | 76,22 / 80,02 | 72,45 / 71,33 | 74,13 / 78,28 |
| Cập nhật từ bằng bố cục | 76,43 / 80,67 | 73,14 / 71,66 | 75,40 / 78,95 |
| Khớp từ trực tiếp và diffusion | 82,70 / 82,27 | 75,15 / 73,59 | 80,11 / 81,38 |
| Cập nhật từ và diffusion | 82,84 / 82,26 | 75,66 / 74,09 | 80,64 / 81,20 |

Cập nhật từ tăng mAP ngay cả khi chấm khớp từ trực tiếp, trên cả ba lượt đầy đủ. Với Ext-2, phép đo chẩn đoán theo lớp trên pilot ban đầu dự đoán đúng 1.126 trong 1.536 quyết định giữa lớp đúng và lớp cạnh tranh. Sau cập nhật số đúng là 1.170. Nếu giữ nguyên đóng góp của tất cả từ trùng chính tả với tên lớp kiểm thử và chỉ thay đóng góp của các từ khác, số đúng là 1.162. Phép phân tích tách riêng đóng góp cộng vào biên điểm đã tính. Nó dùng nhãn sau khi xếp hạng và chưa loại trừ tác dụng của từ đồng nghĩa hay các tên lớp nằm sẵn trong từ điển. [Kiểm tra từ theo lớp](word-attribution.json).

Khi dùng cùng một danh từ trần cho cả sketch và ảnh, mAP khớp từ trực tiếp tăng trên cả hai pilot và toàn bộ TU. Trên TU nó tăng từ 71,79 lên 73,77. Khi thêm diffusion, ảnh hưởng trên Ext-1 kém ổn định. [Đối chứng bỏ mô tả miền](domain-name-ablation.md).

## Độ mới và mốc so sánh

Khớp hình học và dùng ảnh đã xác minh để mở rộng truy vấn đã có tiền lệ. Phần có thể phát triển là cách dùng **mức tăng về tần suất của một khái niệm trong nhóm ảnh được cấu trúc truy vấn ủng hộ, so với chính nhóm ảnh mà CLIP đã đề cử**. Nó làm thay đổi cách diễn giải sketch trong không gian từ trước khi xếp ảnh. Kiểm soát cùng ngân sách, việc trộn trực tiếp từ của ảnh đã xác minh đạt mAP thấp hơn. Việc ghép các thành phần với bộ điểm vùng mạnh vẫn cần được phân biệt với lợi ích của cơ chế riêng. [Tiền lệ và thiết kế đối chứng](prior-art-controls.md).

C-DisAmb báo cáo 81,2 mAP@200 và 78,9 P@200 trên Sketchy bằng prompt fine-tuning. Mốc training-free ở đây còn cách 3,06 điểm mAP và 2,90 điểm P trên Ext-2. Giao thức chọn tham số và cách dùng dữ liệu khác nhau. Kết quả hiện tại là bằng chứng thực nghiệm cho một cơ chế còn cần phát triển để nhắm hội nghị A/A*. [Zhou và cộng sự, CVPR 2026, bảng 1](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhou_Modeling_the_Visual_Ambiguity_of_Human_Sketches_CVPR_2026_paper.pdf).

[Số liệu ngoài cả hai pilot](replication-audit.json) · [Độ nhạy theo lớp trên TU](tuberlin-class-deltas.json) · [Tensor từng truy vấn Ext-1](full-sketchy_1-t4.zip) · [Ext-2](full-sketchy_2-t4.zip) · [TU](full-tuberlin-t4.zip).
