# Hai khái niệm bổ sung cho truy hồi ảnh xuyên miền

**Câu hỏi nghiên cứu.** Khi ảnh truy vấn và ảnh đích khác miền, bằng chứng từ hai phía có nhất thiết phải hội tụ vào một tên gọi? Thử nghiệm trước, với cùng từ điển nguyên văn và cùng toán tử áp dụng cho mọi dataset, cho thấy dùng hai từ độc lập tốt hơn buộc chúng giống nhau. Điều này gợi ý một giả thuyết có thể kiểm tra: giảm khả năng hai đường suy luận đếm lại đúng một tên, trong khi giữ nguyên xác suất của mọi từ trên từng đường. [Nguồn của giả thuyết](https://github.com/HieuNM1804/training-free-ZS-SBIR/blob/2d6cbd9/docs/results/domain-agnostic-word-feedback/README.md)

## Phương pháp

1. **Tạo chỉ mục đích.** Mã hóa ảnh và 20.931 tên gọi công khai bằng CLIP ViT-B/32 đóng băng. Gallery không nhãn cung cấp phân bố từ cho từng ảnh và các ảnh lân cận. Toàn bộ miền dùng chung biểu diễn văn bản nguyên văn, nhiệt độ từ checkpoint và một lượt gộp qua mọi ảnh khác. Không nhập tên miền hay tham số dò trên một split.
2. **Tách hai nguồn.** Với ứng viên $i$ có phân bố từ $b$, nguồn đầu là likelihood $l_w$ từ truy vấn. Nguồn hai là mức các ảnh đích **khác** hỗ trợ từ $v$ phù hợp với chính truy vấn này, ký hiệu $h_i(v)$. Điểm với hai từ độc lập là $\mathbb E_b[l]\mathbb E_b[h_i]$.
3. **Ghép cặp có kiểm soát.** Xét hai lần lấy mẫu từ cho mỗi nguồn theo $b$. Trong hai cách ghép, chọn cách có ít cặp trùng tên hơn và chia đều khi hai cách hòa. Mỗi từ đã lấy vẫn tham gia đúng một lần nên cả hai phân bố biên giữ nguyên. Công thức giải tích tính kỳ vọng trên toàn bộ từ vựng. Suy luận có kết quả tất định, xử lý từng truy vấn độc lập, đủ để giải thích ảnh nào đổi hạng khi bỏ một từ của truy vấn.
4. **Ví dụ tối giản.** Khi hai từ ngang xác suất, đối chứng độc lập trùng tên 50% và cơ chế này trùng 37,5%. Nếu một nguồn chỉ ủng hộ từ thứ nhất và nguồn kia chỉ ủng hộ từ thứ hai, điểm là 0,3125 thay cho 0,25. Nếu cả hai cùng ủng hộ từ thứ nhất, điểm là 0,1875. Số liệu benchmark quyết định liệu sự thay đổi này có ích cho truy hồi.

Sơ đồ xác suất có biểu thức chỉ gồm một đường chéo và bốn tích hạng một. Nhờ đó chấm được mọi từ mà không phải duyệt từng cặp trong hơn 438 triệu cặp từ có thứ tự cho mỗi ảnh. Điểm bậc hai theo likelihood của query cho phép tính chính xác tác động qua cả hai đường khi vô hiệu hóa một từ. [Công thức và thiết kế trước lượt chạy](../../COMPLEMENTARY_WITNESS_PROTOCOL.md) · [Mã tính điểm và giải thích](../../../tf_sbir/complementary_concepts.py)

## Thực nghiệm

Mỗi ô ghi mAP / precision (%). Sketchy Ext‑1 và TU-Berlin 30 lớp dùng mAP@all / P@100. Sketchy Ext‑2 dùng mAP@200 / P@200.

| Cấu hình | Ext-1 | Ext-2 | TU-Berlin |
|---|---:|---:|---:|
| Khớp tuyến tính | 76,20 / 78,10 | 71,17 / 69,88 | 75,06 / 76,64 |
| Một từ dùng cho cả hai phía | 78,03 / 78,49 | 71,18 / 70,14 | 76,94 / 76,94 |
| Hai từ độc lập | 78,62 / 79,19 | 72,03 / 70,74 | 77,41 / 77,67 |
| **Ghép để giảm trùng, giữ nguyên hai phân bố biên** | 78,68 / 79,28 | 72,07 / 70,79 | 77,45 / 77,70 |
| Ghép ngược để tăng trùng, cùng hai phân bố biên | 78,56 / 79,12 | 71,98 / 70,71 | 77,38 / 77,62 |
| Loại toàn bộ cặp trùng tên | 78,93 / 79,49 | 72,09 / 70,86 | 77,56 / 77,71 |
| Hai từ độc lập với phân bố biên sau khi loại trùng | 78,84 / 79,31 | 72,08 / 70,82 | 77,52 / 77,67 |

Trên Ext‑1, Ext‑2 và TU, cơ chế chính tăng lần lượt +2,48, +0,90 và +2,39 điểm mAP so với cách khớp tuyến tính cùng đầu vào. Phần lớn mức tăng đã xuất hiện khi dùng hai từ độc lập. Giữ nguyên cả hai phân bố biên, lợi ích riêng của cách ghép chỉ là +0,061 (CI 95% theo lớp từ +0,014 đến +0,105), +0,044 (từ −0,069 đến +0,150) và +0,034 (từ +0,014 đến +0,054) điểm. Chiều tăng xuất hiện ở 22/25, 15/21 và 22/30 lớp. Chênh lệch trên Ext‑2 chưa tách khỏi 0.

Xác suất trùng tên trên gallery giảm từ 9,98% xuống 5,58%, từ 11,19% xuống 6,25% và từ 5,52% xuống 2,33%. Cách ghép ngược có mAP thấp hơn đối chứng độc lập cả ba lần. Cấm trùng hoàn toàn cho điểm cao hơn cơ chế chính ở cả ba split, đồng thời làm thay đổi tần suất từng từ. Khi giữ đúng hai phân bố biên mới và chỉ bỏ ràng buộc cặp, chênh lệch mAP của phép cấm trùng còn +0,094, +0,008 và +0,044 điểm. Dữ liệu phù hợp với một hiệu ứng phụ thuộc nhỏ giữa hai nguồn. Làm dịch chuyển phân bố từ cũng giải thích một phần điểm của phép cấm trùng.

Tại mỗi split, so công thức với mọi tổ hợp $9^4=6.561$ của bốn lần rút, trên bốn phân bố từ lấy từ gallery thật. Sai khác lớn nhất của trọng số cặp là 1,39 × 10⁻¹⁶ và của phân bố biên là 5,56 × 10⁻¹⁷. Với một truy vấn thật và 72 ảnh đích định trước trên mỗi split, chấm lại đủ 20.931 phép bỏ từng từ. Trong tổng số 62.793 can thiệp, không có lần nào sai dấu chênh lệch của cặp ảnh ngoài ngưỡng số học. Sai số chênh lệch tối đa là 1,22 × 10⁻⁵ so với điểm ảnh đầu ban đầu. Việc này kiểm tra khả năng giải thích của mã trên các trường hợp thử. Các phép chấm mAP/precision dùng toàn gallery.

## Đánh giá tính mới và mức độ sẵn sàng

Ba nguồn cơ chế trực tiếp là FreeDom cho vốn từ công khai, relevance models cho phản hồi dựa trên gallery và minibatch optimal transport cho cách ghép hai mẫu. Phần đề xuất là chi phí trùng từ cho hai vai trò của cùng một ứng viên, kỳ vọng có công thức đóng và phép can thiệp vào từng từ có thể kiểm chứng. Việc ghép cặp hay bảo toàn phân bố biên tự thân đã có tiền lệ. Hai tên khác nhau có thể gần nghĩa. Phương pháp hiện đo việc trùng danh tính từ. [FreeDom, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Efthymiadis_Composed_Image_Retrieval_for_Training-Free_Domain_Conversion_WACV_2025_paper.html) · [Lavrenko và Croft, SIGIR 2001](https://www.inf.ed.ac.uk/publications/report/1190.html) · [Fatras et al., AISTATS 2020](https://proceedings.mlr.press/v108/fatras20a.html)

Đây là ứng viên nghiên cứu có một câu hỏi xuyên miền, một can thiệp giữ chính xác phân bố biên và đối chứng đảo chiều. Ưu điểm hiện tại là có thể giải thích và kiểm định cơ chế. Hiệu ứng về mAP của riêng phép ghép còn nhỏ. Baseline lịch sử dùng prompt miền và graph có tham số chọn trên Sketchy đạt 81,86 / 74,19 / 79,16 mAP, cao hơn cấu hình tổng quát hiện tại trên cả ba split. Khoảng cách gồm nhiều thay đổi cùng lúc. Ablation ở nhánh trước phân biệt ảnh hưởng của prompt, graph và cơ chế xác nhận. Một tuyên bố A*/Q1 về độ chính xác hoặc chuyển sang nhiều cặp miền cần thêm bằng chứng. [So sánh prompt và graph ở cùng backbone](https://github.com/HieuNM1804/training-free-ZS-SBIR/blob/2d6cbd9/docs/results/domain-agnostic-word-feedback/README.md) · [Cách chứng nhận thứ hạng trên cơ chế cùng từ](https://github.com/HieuNM1804/training-free-ZS-SBIR/blob/4e4ee20/docs/results/word-deletion-certificates/README.md)

Dữ liệu được duyệt là Sketchy và TU-Berlin. Cả hai đo sketch → photo. Việc quan sát cùng một hiệu ứng trên ba split kiểm tra tính chuyển giữa các bộ dữ liệu của cặp miền đó, chưa chứng minh khả năng chuyển sang một cặp miền khác. Các điểm lịch sử đã có ở thời điểm hình thành giả thuyết. Cấu hình chính của nhánh này đã khóa trước cả ba số đo mới. Chạy toàn bộ trên Colab L4 với hàm KD-SBIR gốc. [Protocol](protocol.json) · [Khóa trước TU](freeze.json) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU](tuberlin.json) · [Dữ liệu từng query](per-query.zip)
