# Chênh lệch prompt mà gallery ảnh có thể phân biệt

Với một sketch, thay cụm “bản vẽ của” bằng “ảnh chụp của” làm thay đổi điểm của từng tên công khai. Phương pháp tính phần thay đổi có thể suy ra từ phân bố ảnh mà những tên đó truy hồi. Các tên đưa đến cùng tập ảnh có thể khác nhau về mặt chữ nhưng cung cấp ít bằng chứng để đổi thứ tự ảnh. Cấu hình chung được chọn trên toàn bộ Sketchy dùng hệ số +1,2 cho phần gallery phân biệt được, −0,6 cho phần còn lại và cập nhật nhánh toàn ảnh. [Cơ chế và tiền lệ SIGIR](../../GALLERY_VISIBLE_DEPICTION.md) · [Kết quả mẫu cân bằng](PILOT_VI.md)

| mAP / precision (%) | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin 30 lớp |
|---|---:|---:|---:|
| Mốc vùng và từ công khai | 84,37 / 83,85 | 77,01 / 74,84 | 80,07 / 80,92 |
| Phân tách theo gallery, cấu hình chung | 84,29 / 83,53 | 77,81 / 75,74 | 79,65 / 80,43 |
| Đối chứng chỉ dùng một tỷ lệ, cùng tham số | 84,31 / 83,88 | 76,60 / 74,40 | 80,30 / 80,90 |
| Xáo liên kết giữa từ và ảnh, cùng tham số | 84,12 / 83,49 | 77,03 / 74,92 | 79,97 / 80,67 |
| Ngoại suy prompt trực tiếp với hệ số chọn riêng | 84,20 / 83,52 | 77,39 / 75,23 | 79,80 / 80,72 |
| Chỉ cập nhật theo các vùng | 84,35 / 83,65 | 77,31 / 75,22 | 79,73 / 80,75 |
| Cập nhật toàn ảnh và các vùng | 84,08 / 83,16 | 77,86 / 75,86 | 78,94 / 79,95 |

Trên Ext-2, phương pháp sửa 219 lỗi ở vị trí đầu và làm mất 162 kết quả đúng ở vị trí đó. So với mốc, chênh lệch là +0,79 điểm mAP, với khoảng tin cậy bootstrap 95% theo lớp [−0,33, +2,61]. Trên Ext-1 chênh lệch là −0,08 [−0,63, +0,48]. Trên TU-Berlin chênh lệch là −0,43 [−1,05, +0,18]. Về cơ chế, Ext-2 ủng hộ việc phân biệt các từ qua chính các ảnh mà chúng truy hồi. Cùng phân tách và hệ số gây thiệt hại trên TU-Berlin. Cấu hình riêng của một split chưa đủ để chọn đây làm phương pháp chung.

Cả ba cột dùng cùng OpenAI CLIP ViT-B/32 đóng băng, gallery không nhãn, từ điển ngoài Open Images, một sketch độc lập mỗi lần. Hai split Sketchy đầy đủ được dùng để thăm dò và chọn theo trung bình mAP, rồi khóa trước TU-Berlin. Chín điều kiện cuối được chấm bằng hàm KD-SBIR gốc trên 14.179, 12.694 và 2.130 sketch. Ext-1 và TU dùng mAP@all và P@100, Ext-2 dùng mAP@200 và P@200. [Khóa](freeze.json) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU](tuberlin.json) · [Kết quả từng sketch](per-query.zip)
