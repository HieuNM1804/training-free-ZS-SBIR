# Chi phí ép một cặp sketch–ảnh khớp nhau

Ép ảnh và sketch khớp nhau làm thay đổi nghiệm căn chỉnh bộ từ neo. Chi phí tối ưu theo cặp có chứa tín hiệu về sự tương ứng giữa hai miền, nhưng xếp hạng kém hơn cả Dot và ánh xạ cố định trên cả ba split. Khác biệt về toán học và tốc độ giải đã được kiểm chứng. Kết quả truy hồi chưa ủng hộ việc chọn phương pháp làm đóng góp khoa học chính.

## Phương pháp và kết quả chính

Mỗi sketch và ảnh cho một posterior trên 601 từ. Trong từng miền, dùng posterior lấy trung bình các vector văn bản đã trừ tâm, rồi chuẩn hóa thành hướng đơn vị $x,y$. Với mỗi cặp, tính mức tăng nhỏ nhất của lỗi căn chỉnh 601 cặp từ khi buộc phép biến đổi trực giao đưa $y$ đến $x$. Xếp ứng viên theo chi phí tăng dần. Thuật toán cần một SVD của ma trận neo cho toàn bộ từ điển, sau đó chấm các cặp bằng tích phân số. [Công thức, suy dẫn và độ phức tạp](../../forced-pair-alignment.md) · [Đối chiếu prior art](../../forced-pair-prior.md)

| Phương pháp | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin 30 lớp |
|---|---:|---:|---:|
| Dot đủ 601 từ | 75,35 / 77,10 | 72,66 / 71,08 | 73,46 / 73,98 |
| Chi phí tối ưu theo cặp | 65,23 / 69,53 | 63,45 / 61,25 | 66,45 / 70,77 |
| Ánh xạ Procrustes cố định | 68,80 / 71,28 | 64,11 / 62,63 | 69,95 / 71,38 |
| Bỏ tương tác theo cặp trong tích phân | 63,90 / 69,23 | 65,63 / 63,21 | 64,18 / 70,57 |
| Tráo trị kỳ dị giữa các trục | 56,93 / 64,76 | 59,52 / 55,19 | 59,38 / 66,63 |
| Chỉ giữ dạng song tuyến tính | 44,21 / 44,42 | 43,20 / 40,15 | 56,52 / 57,79 |

Mỗi ô là mAP / precision (%). Ext-1 và TU-Berlin dùng mAP@all / P@100, Ext-2 dùng mAP@200 / P@200. Chấm đủ 14.179 / 12.694 / 2.130 sketch trên 17.100 / 12.552 / 27.262 ảnh bằng KD-SBIR. Tham số khóa trước cả ba split. Toàn bộ thực nghiệm dùng CLIP ViT-B/32 đóng băng, seed 42 và Colab T4. Commit lượt chính là 9f55c1e. [Cấu hình](freeze.json) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU-Berlin](tuberlin.json) · [Kết quả từng sketch](paired-T4.zip)

## Thành phần nào có bằng chứng?

Lấy cùng số ứng viên mới vào top-K so với Dot cho mỗi sketch. Khi so với đối chứng tráo phổ, tỷ lệ đúng lớp của nhánh tối ưu tăng trung bình theo lớp **+5,11 / +7,56 / +4,28 điểm**. Cả ba khoảng bootstrap 95% theo lớp đều nằm trên 0. Khi so với đối chứng bỏ tương tác, chênh tương ứng là **+0,13 / −1,84 / +0,10 điểm** và cả ba khoảng chứa 0. Phổ gắn đúng trục có ích cho họ phương pháp này. Phần tương tác theo cặp chưa cho thấy lợi ích ổn định.

Để kiểm tra các ảnh khó phân biệt, trong top-K của Dot, ghép từng cặp không giao nhau ở hai hạng liền kề theo điểm đối chứng. Chỉ tính cặp gồm một ảnh đúng và một ảnh sai mà nhánh tối ưu thực sự đảo thứ tự. Bảng ghi **tỷ lệ đảo đúng khi gộp mọi cặp** và **chênh lệch tỷ lệ đúng giữa hai phương pháp** khi tính từng sketch có cặp đảo, rồi lấy trung bình trong từng lớp và giữa các lớp. Hai mẫu số khác nhau. Mốc ngang nhau lần lượt là 50% và 0 điểm.

| Đối chứng và đại lượng | Ext-1 | Ext-2 | TU-Berlin |
|---|---:|---:|---:|
| Bỏ tương tác, tỷ lệ đảo đúng gộp theo cặp | 55,98% | 51,02% | 57,10% |
| Bỏ tương tác, chênh trung bình theo lớp | −11,52 | −7,04 | −22,77 |
| Ánh xạ cố định, tỷ lệ đảo đúng gộp theo cặp | 44,86% | 50,00% | 50,32% |
| Ánh xạ cố định, chênh trung bình theo lớp | +16,75 | +11,70 | +14,44 |

Với ánh xạ cố định, khoảng bootstrap 95% của chênh theo lớp là **[+5,99 đến +27,60] / [−0,32 đến +23,05] / [−6,76 đến +34,96]**. Chỉ Ext-1 cho thấy lợi ích theo lớp rõ ràng trên nhóm cặp được chọn. Khi xét tất cả truy vấn và toàn thứ hạng theo giao thức, tổng điểm mAP của nhánh tối ưu thấp hơn ánh xạ cố định ở cả ba split. [Phép so với ánh xạ cố định](representation-audit/)

## Hai đối chứng bổ sung và kiểm tra sai số

| Đối chứng | Ext-1 | Ext-2 | TU-Berlin |
|---|---:|---:|---:|
| Phép khả thi tốt hơn giữa quay và phản xạ | 66,27 / 70,61 | 64,42 / 62,38 | 67,55 / 71,23 |
| Chi phí dịch chuyển có trọng số theo phổ | 62,93 / 67,07 | 60,93 / 58,32 | 64,84 / 69,44 |
| Cosine trên posterior 601 từ | 72,47 / 73,89 | 67,25 / 66,12 | 70,78 / 70,96 |
| Cosine trên posterior đã trừ tâm đều | 73,68 / 74,65 | 68,01 / 67,06 | 72,10 / 71,40 |
| Dot trên vector neo đã trừ tâm trong cơ sở Procrustes | 66,74 / 70,88 | 70,15 / 66,69 | 66,75 / 73,96 |
| Ánh xạ cố định, nhân lại độ dài posterior gốc | 64,15 / 71,13 | 70,72 / 65,67 | 62,45 / 70,36 |

Nghiệm tối ưu đạt chi phí thấp hơn hoặc bằng phương án khả thi trên mọi cặp trong dung sai $2\times10^{-4}$. Phương án khả thi đơn giản lại đạt cả hai metrics cao hơn trên cả ba split. Dot cao hơn cosine trên posterior ở cả ba, cho thấy việc chuẩn hóa đã làm mất tín hiệu xếp hạng. Nhân lại độ dài posterior cải thiện ánh xạ cố định trên Ext-2 về mAP nhưng giảm trên hai bộ còn lại. Việc tối ưu đúng mục tiêu hình học chưa đủ để suy ra một phép so tốt cho truy hồi theo lớp. [Đối chứng khả thi](feasible-audit/) · [Đối chứng biểu diễn](representation-audit/)

Với 64 truy vấn chọn theo seed 42 cho mỗi full gallery, tích phân 32 nút và 64 nút FP64 khớp top-1 lẫn tập ảnh trong top-K ở **192/192 truy vấn**. Sai khác điểm tuyệt đối lớn nhất dưới $4\times10^{-5}$. So thứ tự 12 cặp gần hòa trong top-32 với SVD trực tiếp FP64, bản 32 nút đúng 11/12, bản 64 nút đúng 12/12. Mọi sketch và ảnh có độ dài vector neo đã trừ tâm lớn hơn 0,095 lần chuẩn RMS của bộ neo tương ứng. Giả thuyết suy biến do posterior đều không giải thích được mức sụt mAP quan sát. Cả hai benchmark chỉ kiểm tra chiều sketch → photo. [Kiểm tra số và posterior](numerical-audit.json)
