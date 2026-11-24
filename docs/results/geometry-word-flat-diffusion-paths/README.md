# Hai lần dùng diffusion có ảnh hưởng khác nhau đến truy hồi

Cả bốn hàng đều giữ GCWE, khớp vùng và khớp từ toàn ảnh. Chỉ thay phép lan truyền bằng toán tử đồng nhất ở một hoặc cả hai nơi. Điểm toàn ảnh bao gồm phần hiệu chỉnh của GCWE. Ở nhánh vùng, diffusion tác động lên trọng số từ của gallery trước bước chọn vùng.

| Diffusion được bật | Ext-1 mAP / P@100 | Ext-2 mAP@200 / P@200 | TU mAP / P@100 |
|---|---:|---:|---:|
| Cả hai | 84,78 / 84,24 | 77,84 / 75,66 | 81,49 / 82,20 |
| Chỉ toàn ảnh | 83,75 / 84,20 | 76,98 / 75,19 | 80,73 / 82,09 |
| Chỉ vùng | 81,20 / 83,73 | 76,38 / 74,81 | 78,48 / 80,72 |
| Tắt cả hai | 80,29 / 83,61 | 76,10 / 74,44 | 77,78 / 80,48 |

Khi nhánh vùng đã bật, diffusion toàn ảnh tăng 3,582 / 1,459 / 3,016 mAP, nhưng có 785 / 1.025 / 119 truy vấn mất ảnh đúng cuối cùng khỏi top 200; chỉ 5 / 3 / 5 truy vấn tìm lại được một ảnh đúng. Khi nhánh toàn ảnh đã bật, bật diffusion vùng tăng 1,030 / 0,864 / 0,763 mAP và tương ứng có 186 / 296 / 32 lượt mất, 27 / 41 / 5 lượt tìm lại. Tổn thất độ bao phủ lớn hơn ở đường toàn ảnh trên cả ba split. Bỏ hẳn đường này làm giảm mAP; giữ cấu hình hai đường làm mốc. [Đối chiếu top-1 và top-200](coverage.json)

[Protocol](protocol.md) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU](tuberlin.json) · [Theo truy vấn](paired-L4.zip)
