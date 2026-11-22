# Thay trạng thái của danh từ giữa hai prompt trong CLIP

Giữ nguyên các token của tên vật, chạy hai câu mô tả tới cùng một block, rồi thay các trạng thái của riêng tên từ câu “sketch” sang câu “photo” và ngược lại. Cho CLIP chạy tiếp để đo phần chênh lệch đi qua tên và phần còn lại đi qua các token khác. Hai thành phần cộng lại đúng bằng chênh lệch giữa hai cách mô tả. Phương pháp chỉ thay bảng biểu diễn văn bản trước khi chấm ảnh. [Giao thức và nguồn](../../DEPICTION_TOKEN_PATHS.md)

| mAP / precision (%) | Sketchy Ext-1 | Sketchy Ext-2 |
|---|---:|---:|
| Mốc từ và vùng | 84,64 / 83,96 | 80,63 / 78,38 |
| Can thiệp hai chiều vào các token danh từ | 84,75 / 84,06 | 81,18 / 78,71 |
| Can thiệp một chiều, cùng hệ số | 84,77 / 84,12 | 80,98 / 78,51 |
| Cùng độ dài hướng chỉnh sửa, song song với chênh lệch cả prompt | 84,93 / 84,34 | 79,38 / 77,56 |
| Lấy hướng của một tên khác cùng độ dài token | 84,76 / 83,95 | 80,48 / 78,33 |
| Chỉ ngoại suy cả prompt, hệ số chọn riêng | 84,40 / 83,65 | 81,09 / 78,87 |
| Can thiệp một chiều, chọn hệ số và nhánh riêng | 84,78 / 84,30 | 81,05 / 78,62 |

Cấu hình chính trừ bốn lần phần thay đổi đi qua danh từ sau block 11. Nó sửa toàn sketch lẫn các vùng, và giữ nguyên phần đóng góp của các token còn lại. Ở Ext-2, nó hơn đối chứng đổi sang danh từ khác 0,70 điểm mAP với cùng hệ số. So với nền, chênh lệch là +0,11 điểm trên Ext-1 và +0,55 trên Ext-2. Khoảng tin cậy bootstrap 95% theo lớp lần lượt là [−0,30, +0,51] và [−0,14, +1,27]. Đây là bằng chứng thăm dò cho việc kiểm tra nguồn của thay đổi, với độ bất định còn lớn. Từ 20.931 tên công khai, sai lệch lớn nhất của phép cộng lại hai thành phần dưới 0,00000002 trong không gian biểu diễn.

Mỗi lớp có 36 sketch, tương ứng 900 và 756 truy vấn, với 17.100 và 12.552 ảnh đầy đủ. Các điều kiện cuối trong bảng được chấm bằng hàm KD-SBIR gốc. [Cấu hình và so sánh ghép cặp](pilot.json) · [Điểm từng truy vấn](pilot-per-query.zip) · [Ext-1](pilot-sketchy_1.json) · [Ext-2](pilot-sketchy_2.json)
