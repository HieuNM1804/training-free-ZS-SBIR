# Kiểm tra hai lần đọc truy vấn qua liên thông gallery

Mỗi ảnh có điểm Dot từ lần đọc sketch gốc và lần đọc sketch lật ngang. Ở một ngưỡng, giữ những ảnh đạt ngưỡng theo ít nhất một lần đọc. Ảnh ứng viên được xác nhận khi thành phần liên thông của nó chứa bằng chứng cho cả hai lần đọc, có thể đến từ hai ảnh khác nhau. Vị trí nối hai loại bằng chứng thay đổi mức xác nhận ngay cả khi mọi ảnh giữ cùng điểm Dot trung bình. [Công thức, phép dựng kiểm chứng và prior art](protocol.md)

## Kết quả full unseen

| Cấu hình | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin 30 lớp |
|---|---:|---:|---:|
| Dot đủ 601 từ | 75,35 / 77,10 | 72,66 / 71,08 | 73,46 / 73,98 |
| Nối bằng chứng của hai lần đọc | 75,31 / 77,12 | 73,11 / 70,99 | 73,52 / 73,59 |
| Tráo danh tính hai lần đọc theo ảnh | 75,52 / 77,86 | 73,72 / 71,73 | 73,90 / 74,96 |
| Chỉ lấy marker từ ảnh hỗ trợ cả hai lần đọc | 75,30 / 77,09 | 73,11 / 71,02 | 73,50 / 73,56 |
| RGB diffusion trên cùng graph láng giềng hai chiều | 76,83 / 77,86 | 73,64 / 71,97 | 74,98 / 74,60 |

Các cặp số là mAP / precision (%). Ext-1 và TU-Berlin dùng mAP@all / P@100, Ext-2 dùng mAP@200 / P@200. Chấm 14.179 / 12.694 / 2.130 sketch trên 17.100 / 12.552 / 27.262 ảnh bằng KD-SBIR gốc. Colab T4, OpenAI CLIP ViT-B/32 đóng băng, seed 42, commit 3e3d0b1. Dot được đối chiếu từng truy vấn. Cùng cấu hình đã khóa dùng trên cả ba split. Điểm diffusion ở đây dùng graph cosine 16 láng giềng có điều kiện hai chiều. [Khóa cấu hình](freeze.json) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU-Berlin](tuberlin.json) · [Kết quả từng truy vấn](paired-T4.zip)

## Đối chứng cơ chế

Trong dữ liệu dựng, giữ điểm trung bình, min, max và graph cho mọi ảnh. Tráo hai lần đọc tại ảnh giữa làm mức xác nhận của ảnh đầu đổi từ 0,4 lên 0,8. Trên gallery thật, khi ghép cùng số ảnh mới vào top-K từ mỗi nhánh, tỷ lệ đúng lớp là **67,63 / 69,78**, **59,01 / 60,63**, **65,37 / 68,41** (%) cho hai lần đọc thật so với đối chứng tráo tương ứng. Đối chứng dùng một mặt nạ tráo xác định từ seed 42, giữ nguyên cặp giá trị mỗi ảnh. Chênh lệch theo lớp của bằng chứng thật là −1,52 / −1,06 / −2,10 điểm, với khoảng bootstrap theo lớp nằm dưới 0 cho Ext-1 và TU-Berlin.

Thêm cạnh đổi lần đọc nâng điểm so với marker cùng ảnh trong khoảng 0,45–0,66% số cặp truy vấn–ảnh. Chênh tỷ lệ đúng lớp ở cùng số ảnh mới giữa hai nhánh là +0,06 / −0,05 / −0,01 điểm tính trên ảnh. Cả ba khoảng bootstrap theo lớp đều chứa 0. Bằng chứng trên gallery này chưa xác nhận lợi ích từ việc gắn đúng danh tính hai lần đọc hoặc cho phép nhân chứng nằm ở hai ảnh.

## Giới hạn của khoảng điểm hai lần đọc

Đặt \(l_i=\min(a_i,b_i)\) và \(u_i=\max(a_i,b_i)\). Mức nối bằng chứng luôn thuộc \([l_i,u_i]\). Một điều kiện để có thể sửa top-1 là ảnh đúng tốt nhất theo \(u\) phải có điểm ít nhất bằng ảnh sai cao nhất theo \(l\). Đây là kiểm tra lý tưởng với nhãn, dùng sau khi đã có thứ hạng. Với precision, gán \(u\) cho mọi ảnh đúng và \(l\) cho mọi ảnh sai rồi ưu tiên ảnh đúng khi hòa để lấy cận trên lạc quan. [Kết quả kiểm tra khoảng điểm](interval-audit.json)

| Chẩn đoán | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin |
|---|---:|---:|---:|
| Dot sai top-1 | 4.234 | 4.464 | 604 |
| Sai top-1 chắc chắn với mọi điểm trong khoảng | 2.730 | 2.804 | 380 |
| Lỗi top-1 còn khả năng sửa trong khoảng | 1.504 | 1.660 | 224 |
| Lỗi top-1 được thực nghiệm hiện tại sửa | 218 | 235 | 20 |
| Cận trên lạc quan của precision (%) | 82,17 | 77,83 | 80,72 |

Hơn 60% lỗi top-1 nằm ngoài khả năng sửa của họ phương pháp giữ nguyên khoảng điểm giữa hai lần đọc này. Một phần lỗi còn nằm trong khoảng, và đối chứng cho thấy liên thông của hai lần đọc chưa khai thác được phần đó tốt hơn. Kết luận áp dụng với cặp sketch gốc/lật ngang và hai benchmark sketch → photo đã chấm.

Thời gian chấm GPU của các nhánh cùng lượt là 9,23 / 4,46 / 2,58 giây sau lúc lập graph và có embedding. Hồi phục đạt điểm cố định trong tối đa 48 / 48 / 40 vòng, tính cả kiểm tra theo từng nhóm tám vòng. Thời gian phát lại evaluator KD-SBIR trên CPU đo riêng.
