# Khớp vùng đổi vai trò khi thêm ngữ nghĩa toàn ảnh

So với mỗi ứng viên tự chọn cả kích thước và vị trí, cho các ứng viên chọn kích thước nhưng dùng chung trọng số vị trí làm thay đổi mAP của điểm vùng đứng riêng lần lượt **+0,366 / +0,211 / +1,636** trên Ext-1, Ext-2 và TU. Khi ghép cùng điểm ngữ nghĩa toàn ảnh ở hệ số cũ, đúng thay đổi đó cho **−0,184 / −0,307 / −0,040**. Điều kiện về dấu đã khóa được tái hiện trên cả hai split Sketchy và cùng chiều trên TU. Đặc biệt, mức giảm 0,040 trên TU khi ghép là nhỏ.

| Cấu hình (mAP / precision %) | Ext-1, 7.267 truy vấn | Ext-2, 5.782 | TU, 2.130 |
|---|---:|---:|---:|
| Vùng, mỗi ứng viên chọn kích thước và vị trí | 80,44 / 79,40 | 72,63 / 71,17 | 69,98 / 68,54 |
| Vùng, mỗi ứng viên chỉ chọn kích thước | 80,80 / 79,89 | 72,84 / 71,27 | 71,62 / 70,16 |
| Ngữ nghĩa toàn ảnh + vùng chọn cả hai | 84,31 / 83,79 | 77,60 / 75,40 | 80,57 / 81,47 |
| Ngữ nghĩa toàn ảnh + vùng chỉ chọn kích thước | 84,12 / 83,69 | 77,29 / 75,09 | 80,53 / 81,42 |

Tương tác theo mAP lấy hiệu hai mức thay đổi là −0,550, −0,518 và −1,675 điểm. Khoảng bootstrap 95% lấy trọng số bằng nhau giữa các lớp lần lượt là [−0,910; −0,164], [−1,044; +0,117] và [−2,439; −0,949]. Cải thiện một nhánh đứng riêng không đảm bảo cải thiện khi ghép. Thí nghiệm tắt GCWE để cô lập tương tác. Phân tích này chưa chọn một kiến trúc mới.

[Protocol](protocol.md) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU](tuberlin.json) · [Theo truy vấn](paired-L4.zip)
