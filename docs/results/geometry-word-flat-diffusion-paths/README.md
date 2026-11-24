# Hai lần dùng diffusion có ảnh hưởng khác nhau đến truy hồi

Cả bốn hàng đều giữ GCWE, khớp vùng và khớp từ toàn ảnh. Chỉ thay phép lan truyền bằng toán tử đồng nhất ở một hoặc cả hai nơi. Điểm toàn ảnh bao gồm phần hiệu chỉnh của GCWE. Ở nhánh vùng, diffusion tác động lên trọng số từ của gallery trước bước chọn vùng.

| Diffusion được bật | Ext-1 mAP / P@100 | Ext-2 mAP@200 / P@200 | TU mAP / P@100 |
|---|---:|---:|---:|
| Cả hai | 84,78 / 84,24 | 77,84 / 75,66 | 81,49 / 82,20 |
| Chỉ toàn ảnh | 83,75 / 84,20 | 76,98 / 75,19 | 80,73 / 82,09 |
| Chỉ vùng | 81,20 / 83,73 | 76,38 / 74,81 | 78,48 / 80,72 |
| Tắt cả hai | 80,29 / 83,61 | 76,10 / 74,44 | 77,78 / 80,48 |

Khi nhánh vùng đã bật, diffusion toàn ảnh tăng 3,582 / 1,459 / 3,016 mAP, nhưng có 785 / 1.025 / 119 truy vấn mất ảnh đúng cuối cùng khỏi top 200; chỉ 5 / 3 / 5 truy vấn tìm lại được một ảnh đúng. Khi nhánh toàn ảnh đã bật, bật diffusion vùng tăng 1,030 / 0,864 / 0,763 mAP và tương ứng có 186 / 296 / 32 lượt mất, 27 / 41 / 5 lượt tìm lại. Tổn thất độ bao phủ lớn hơn ở đường toàn ảnh trên cả ba split. Bỏ hẳn đường này làm giảm mAP; giữ cấu hình hai đường làm mốc. [Đối chiếu top-1 và top-200](coverage.json)

### Những truy vấn nào tạo ra thay đổi mAP?

Chia toàn bộ truy vấn theo việc có ít nhất một ảnh đúng trong top 200 trước và sau khi bật diffusion toàn ảnh, giữ diffusion vùng bật. Mỗi cột dưới đây là tổng thay đổi AP trong nhóm chia cho **tổng số truy vấn của split**, tính bằng điểm phần trăm. Tổng bốn cột bằng đúng thay đổi mAP chung trước khi làm tròn.

| Split | Vẫn có ảnh đúng | Mất ảnh đúng cuối cùng | Tìm lại ít nhất một ảnh đúng | Cả hai đều không có ảnh đúng trong top 200 |
|---|---:|---:|---:|---:|
| Sketchy Ext-1 | +3,418 | +0,080 | +0,000 | +0,084 |
| Sketchy Ext-2 | +2,181 | −0,721 | +0,000 | 0,000 |
| TU-Berlin | +3,025 | +0,005 | −0,004 | −0,010 |

Trên Ext-2, 1.025 truy vấn mất ảnh đúng cuối cùng xóa khoảng một phần ba phần tăng +2,181 của nhóm vẫn giữ ảnh đúng. Trên Ext-1, nhóm mất ảnh đúng ở top 200 vẫn đóng góp dương vào mAP@all, dù đóng góp vào P@100 là −0,354 điểm. Hạng của các ảnh còn lại bên dưới ngưỡng 200 cũng ảnh hưởng đến AP@all. Do đó cần theo dõi cả metric gốc và độ bao phủ khi đánh giá một cách sửa cho truy vấn khó. Hai nhóm được xác định bằng nhãn sau khi xếp hạng và không thể dùng trực tiếp để chuyển đổi thuật toán lúc suy luận. [Phân rã đủ bốn đối chứng](coverage-contributions.json)

[Protocol](protocol.md) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU](tuberlin.json) · [Theo truy vấn](paired-L4.zip)
