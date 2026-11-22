# Thay nhóm ảnh tham chiếu và chấm bộ dữ liệu chuyển giao

Cấu hình chính đã khóa chia ngẫu nhiên gallery thành sáu nhóm, chọn 32 từ mỗi nhóm và cộng ba lần log tỷ số giữa hai cách đọc vùng. Cách thứ nhất dùng một nhóm để chọn vùng và nhóm khác để chấm. Cách thứ hai dùng cùng nhóm cho cả hai bước. Điểm sửa đổi được cộng vào mốc khớp từ và vùng đã có. [Cơ chế và thiết kế đối chứng](../../INDEPENDENT_GALLERY_REGION_READOUT.md) · [Khóa trên toàn bộ Sketchy](freeze.json)

## Cùng gallery, ba cách chia độc lập

Các lần chia giữ nguyên sketch, ảnh, checkpoint, tham số và evaluator. Chúng tôi chọn trước hai seed bổ sung, đồng thời lấy trung bình các hiệu chỉnh của cả ba lần chia với trọng số bằng nhau. Việc lấy trung bình diễn ra trước khi xếp hạng.

| mAP / precision (%) trên cùng mẫu 36 sketch mỗi lớp | Ext-1 | Ext-2 |
|---|---:|---:|
| Mốc từ và vùng | 84,64 / 83,96 | 80,63 / 78,38 |
| Lần chia dùng khi phát triển | 84,63 / 84,39 | 81,09 / 78,57 |
| Lần chia thứ hai | 84,69 / 84,51 | 80,61 / 78,43 |
| Lần chia thứ ba | 84,43 / 84,06 | 80,57 / 78,40 |
| Trung bình ba lần sửa điểm | 84,59 / 84,29 | 80,67 / 78,45 |

Trên mẫu Ext-2, lợi ích +0,46 điểm mAP của lần chia ban đầu còn +0,04 điểm khi lấy trung bình ba hiệu chỉnh. Tương quan giữa hai hiệu chỉnh bất kỳ trong 200 ảnh đứng đầu của mốc thay đổi từ 0,59 tới 0,70 ở Ext-2 và từ 0,41 tới 0,55 ở Ext-1. Cả ba seed được chấm bằng KD-SBIR gốc trên 900 và 756 truy vấn. [Bảng điểm và độ tương quan](seed-repeatability.json) · [Điểm từng truy vấn](seed-repeatability.zip)

## Cùng ba ảnh, thay những ảnh còn lại của gallery

Dùng lại đúng 2.231 sketch và 2.205 ảnh chung ở hai split Sketchy. Trên toàn bộ cặp gồm một ảnh đúng và một ảnh sai, cấu hình đã khóa **tăng** tỷ lệ đảo thứ tự giữa hai gallery từ **0,598% lên 1,345%**. Độ trùng của top 100 giảm từ 89,65% xuống 86,28%. Trong tập ảnh cố định là hợp hai top 100 của mốc, chỉ 174 trong số 2.231 sketch có cả ảnh đúng lẫn ảnh sai. Trên 174 sketch đó, tỷ lệ đảo thứ tự là 61,05% ở mốc và 61,01% sau can thiệp. Chênh lệch giữa hai phép đo cho thấy các cặp đứng đầu khó hơn nhiều. Bằng chứng về sự ổn định phụ thuộc vào tập cặp được xét. [Đếm chính xác các cặp đối chứng](shared-scene-intervention.json) · [Số liệu từng sketch](shared-scene-intervention.zip)

## TU-Berlin với cấu hình đã khóa

Trên 2.130 sketch và đủ 27.262 ảnh thuộc 30 lớp, T4 cho **79,53 mAP / 80,30 P@100**. Mốc cùng GPU đạt **80,07 / 80,92**. Chênh lệch −0,55 điểm mAP có khoảng bootstrap 95% theo lớp [−1,15, −0,06]. Phương pháp sửa 18 lỗi hạng nhất và làm sai 56 truy vấn vốn đứng đúng lớp. Đối chứng lấy trung bình các phép đọc cùng nhóm đạt 80,16 / 80,99. Cơ chế tách nhóm chưa đạt tiêu chí làm phương pháp chính. Chọn cấu hình trên hai split Sketchy là thăm dò trên unseen. TU-Berlin chỉ dùng để chấm sau khi khóa. [Số liệu TU-Berlin và đối chứng](tuberlin.json) · [Giao thức lựa chọn](full-protocol.json)
