# Liệu đồng thuận về khái niệm có phát hiện diffusion sai?

Dùng một truy vấn độc lập, CLIP ViT-B/32 đóng băng, 20.931 từ công khai và cùng đồ thị RGB. Hai người láng giềng của một ảnh có thể cùng cho điểm sketch cao vì cùng một từ hoặc vì hai từ khác nhau. Hệ số $R$ đo mức trùng nhau giữa hai phân bố từ khi đã quan sát sketch. Giả thuyết là sự trùng nhau này có thể bảo vệ ảnh đúng khi đồ thị đẩy một ảnh sai lên cao. [Thiết kế trước lượt chạy](../../CONDITIONAL_AGREEMENT_PROTOCOL.md)

| Điểm mAP / precision (%) | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin 30 lớp |
|---|---:|---:|---:|
| Khớp từ trực tiếp | 75,31 / 78,94 | 71,42 / 70,28 | 73,35 / 77,22 |
| Khớp từ + diffusion RGB | 81,86 / 81,33 | 74,19 / 72,68 | 79,16 / 80,24 |
| Đồng thuận theo sketch | 81,33 / 81,23 | 74,26 / 72,86 | 78,11 / 79,46 |
| Đồng thuận so với truy vấn phẳng | 81,83 / 81,12 | 73,74 / 72,30 | 79,03 / 79,05 |
| Gộp căn xác suất từ một lần trên gallery | 82,03 / 81,70 | 74,35 / 72,85 | 79,39 / 80,14 |

Giữa hai ứng viên đã đặt trước, chọn đồng thuận theo sketch bằng trung bình mAP trên hai split Sketchy unseen. Chênh lệch mAP của ứng viên với diffusion lần lượt là −0,53, +0,07 và −1,05 điểm. Các khoảng bootstrap 95% theo lớp đều chứa 0. Lợi ích của ứng viên chưa được xác nhận; đối chứng tính một lần trên gallery có mAP cao hơn ở cả ba split.

Trong số các truy vấn mà diffusion làm mất hạng nhất đúng, hệ số đồng thuận cao hơn ở ảnh đúng ban đầu chỉ trong 27/217, 35/481 và 9/67 trường hợp. Riêng TU, ứng viên cứu 2 trong 67 lượt mất này, đồng thời đảo sai 25 kết quả mà diffusion vừa sửa đúng. Hoán vị danh tính từ ở một nửa gallery cho mAP TU 78,111, sát 78,110 với từ đúng. Kết luận giới hạn ở hệ số được kiểm tra.

Ext-2 dùng mAP@200 và P@200. Hai split còn lại dùng mAP@all và P@100. GPU Colab L4 chấm 14.179, 12.694 và 2.130 truy vấn cho từng cấu hình. Tệp JSON chứa điểm ghép cặp theo lớp, các trường hợp cụ thể và SHA-256 dữ liệu theo truy vấn. Tệp ZIP chứa các tensor để kiểm tra lại. [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU-Berlin](tuberlin.json) · [Lựa chọn trước TU](freeze.json) · [Dữ liệu từng truy vấn](per-query.zip)
