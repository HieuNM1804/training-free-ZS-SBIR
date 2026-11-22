# Thay hoạt hóa neuron bằng quan sát từ patch ảnh đích

Hệ thống dùng OpenAI CLIP ViT-B/32 đóng băng. Gallery của mỗi split cung cấp 192 ảnh để nhận diện neuron nhạy, 768 ảnh cho giá trị neuron và 144 ảnh giữ lại để kiểm tra phép thay. Các nhóm tách biệt và không dùng nhãn. Hai pilot Sketchy unseen chọn cấu hình block 2, tám neuron nhạy với dịch chuyển, 32 patch ảnh gần nhất theo đặc trưng CLIP tầng cuối. Giữ nguyên tham số trên TU-Berlin và dựng bảng từ chính gallery không nhãn của TU. Cấu hình đã được đẩy lên Git trước khi bắt đầu chấm TU. [Cấu hình khóa](freeze.json) · [Cơ chế, đối chứng và các bài tham khảo](../../SEMANTIC_PATCH_NEURON_REPLACEMENT.md)

## Kết quả trên toàn bộ truy vấn

Mỗi ô ghi mAP / precision (%). Ext-1 dùng 14.179 sketch × 17.100 ảnh, mAP@all / P@100. Ext-2 dùng 12.694 × 12.552, mAP@200 / P@200. TU-Berlin dùng 2.130 × 27.262 ảnh, đủ 30 lớp, mAP@all / P@100. Trên cùng T4, sai số tuyệt đối tối đa giữa phép tính theo thứ hạng và evaluator KD-SBIR gốc là 2,39 × 10⁻⁷ ở từng truy vấn.

| Phương án | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin |
|---|---:|---:|---:|
| Nền vùng, từ công khai và diffusion RGB | 84,37 / 83,85 | 77,02 / 74,83 | 80,07 / 80,92 |
| Thay neuron bằng giá trị từ patch ảnh tương đồng | 84,27 / 83,70 | 77,04 / 74,76 | 80,03 / 81,07 |
| Thay bằng trung bình gallery, tự chọn cấu hình | 84,43 / 83,94 | 77,04 / 74,80 | 80,02 / 80,80 |
| Tráo ghép patch với giá trị neuron, tự chọn cấu hình | 84,44 / 83,95 | 77,02 / 74,80 | 80,02 / 80,79 |
| Chỉ cosine CLIP gốc | 30,29 / 41,63 | 36,35 / 33,85 | 29,07 / 46,27 |
| Chỉ cosine sau thay neuron trên sketch | 30,13 / 41,40 | 36,37 / 33,88 | 29,36 / 46,68 |

Chênh mAP của phương án chính so với nền là −0,100 điểm trên Ext-1, +0,019 trên Ext-2 và −0,046 trên TU. Khoảng tin cậy 95% bootstrap ghép cặp theo lớp lần lượt là [−0,26; +0,05], [−0,18; +0,26] và [−0,34; +0,20]. Các cặp top-1 được sửa / bị hỏng là 79/93, 101/107 và 21/23. Việc chọn đúng patch cung cấp hoạt hóa chưa thể giải thích mức tăng ổn định. So với mốc có huấn luyện C-DisAmb 81,2 mAP@200, Ext-2 còn khoảng 4,16 điểm. Thiết lập chọn tham số trên unseen và bảng gallery bổ sung cần được ghi kèm khi so với paper. [Bảng 1 của C-DisAmb, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhou_Modeling_the_Visual_Ambiguity_of_Human_Sketches_CVPR_2026_paper.pdf)

Trên pilot cỡ 900 / 756 truy vấn, điểm mAP của phương án chính là 84,77 / 80,81, so với 84,64 / 80,64 của nền. Chênh nhỏ này đã co lại khi chấm toàn bộ sketch. Dùng đúng nhánh từ điển 20.931 từ của nền, thay riêng đầu ra toàn ảnh với hệ số gốc 0,65, cũng chỉ đạt mAP 84,71 / 80,81 trên pilot. [Đối chứng toàn từ vựng](original-readout)

## Sai số nằm ở đâu?

- **Cùng một vật, các bản vẽ có thể cho thứ hạng rất khác.** Đối với nền vùng, trong các top-1 sai có ít nhất hai bản vẽ xác minh được cùng ảnh nguồn, 2.136/2.320 ở Ext-1 và 2.893/3.212 ở Ext-2 có một bản vẽ khác truy hồi đúng. Khi tráo bản vẽ trong cùng lớp, tỷ lệ này còn cao hơn (98,73% và 95,65%), do nhiều đối tượng cũng có bản vẽ dễ. Số liệu thuyết phục hơn là phân rã phương sai AP sau khi cố định lớp. Phần biến thiên giữa các bản vẽ của cùng đối tượng chiếm 75,7% [69,7%; 81,4%] và 83,1% [78,3%; 87,7%]. Trong hai bản vẽ sai của cùng đối tượng, tỷ lệ cùng nhầm một lớp là 30,6% / 44,2%, cao hơn 20,9% / 38,1% sau khi tráo trong lớp. [Dữ liệu và cách phân rã](drawing-error-components.json)
- **Lỗi có thể xem lại trên dữ liệu gốc.** Hai bản vẽ từ ảnh gấu mèo `n02508021_12996` cho AP@200 lần lượt 99,21% và 0%. Bản thứ nhất truy hồi ảnh gấu mèo ở hạng đầu, bản thứ hai truy hồi ảnh lớp chuột. Với ảnh xe tăng `n04389033_7763`, hai bản cho AP 99,61% và 15,19%; hạng đầu của bản thứ hai là xe lăn. Đây là ví dụ minh họa, chọn ở giữa phân bố chênh AP trong các lớp có nhiều đối tượng với cả hai kết quả. [Đường dẫn ảnh và sketch cụ thể](drawing-error-components.json)
- **Giới hạn của can thiệp hiện tại.** Ngay cả khi biết AP thật và chọn đúng cho từng truy vấn giữa nền và cấu hình chính, mức tăng mAP tối đa trên ba split mới lần lượt là +0,465, +0,814 và +0,515 điểm. Chọn trong toàn bộ nhóm đối chứng đã khóa chỉ nâng các mức trần lên +1,013, +1,574 và +0,911. Đây là trần thực nghiệm của danh sách cấu hình cụ thể, có dùng nhãn hậu nghiệm. [Mức trần theo truy vấn](ceiling-full.json)
- **Phần giống nhau giữa các bản vẽ không tự chọn được cấu hình tốt.** Lấy điểm can thiệp của các bản vẽ khác cùng ảnh nguồn như một chẩn đoán hậu nghiệm, lựa chọn nhị phân chỉ tăng khoảng 0,01 điểm mAP trên mỗi split đầy đủ. Ở phép đo khác, trong pilot, tỷ lệ ảnh cho patch có cùng nhãn với sketch cao hơn tỷ lệ ngẫu nhiên. Sau kiểm soát lớp và độ khó ban đầu, việc có nhiều ảnh cùng lớp hơn không dự báo lợi ích dương ổn định so với đối chứng tráo nguồn cho giá trị. [Độ lặp lại giữa bản vẽ](choice-repeatability-full.json) · [Chẩn đoán ảnh cho patch ở Ext-2](donor-diagnosis-pilot-official-sketchy_2.json)

Các nhãn ảnh, ảnh nguồn và bản vẽ liên quan chỉ tham gia chẩn đoán sau xếp hạng. Mọi điểm truy hồi báo trong bảng dùng đúng một sketch tại thời điểm suy luận. Hai bộ dữ liệu đều đo chiều sketch → photo. Kết quả hiện tại xác nhận một thuộc tính của các neuron và một giới hạn của cách thay hoạt hóa này; chưa đủ cơ sở để chọn nó làm phương pháp chính cho bài báo.
