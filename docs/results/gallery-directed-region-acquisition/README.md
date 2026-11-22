# Gallery chọn một vùng sketch trước khi CLIP phóng to

Toàn sketch đề xuất 12 nhóm ảnh đang cạnh tranh. Một nửa ảnh trong gallery dự báo thứ tự của từng cặp nhóm theo từng từ có thể xuất hiện ở một vùng. Nửa ảnh còn lại đo xác suất quyết định đó sai. Hệ thống chọn một vùng trước khi đọc sketch ở độ phân giải cao. Các đối chứng một vùng dùng cùng số lượt đọc CLIP.

| mAP / precision (%) | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin 30 lớp |
|---|---:|---:|---:|
| Chỉ đọc toàn sketch | 81,86 / 81,33 | 74,19 / 72,68 | 79,16 / 80,24 |
| Một vùng ngẫu nhiên, kỳ vọng theo hai kích thước | 81,61 / 81,27 | 73,30 / 71,27 | 77,97 / 78,80 |
| Vị trí ngẫu nhiên ở đúng kích thước phương pháp chọn | 82,68 / 82,28 | 75,01 / 72,80 | 78,74 / 79,28 |
| Một vị trí chọn từ gallery và cố định cho mọi sketch | 83,13 / 82,78 | 75,29 / 73,15 | 78,92 / 79,49 |
| Vị trí cố định của gallery trong đúng kích thước đã chọn | 83,14 / 82,78 | 75,29 / 73,15 | 78,95 / 79,49 |
| Chọn cả kích thước và vị trí theo sketch | 82,97 / 82,54 | 75,20 / 73,00 | 78,73 / 79,48 |
| Gộp đủ 13 vùng | 84,00 / 83,45 | 76,43 / 74,17 | 79,50 / 80,12 |
| Chọn một vùng hậu nghiệm bằng nhãn, chỉ để đo dư địa | 88,68 / 88,65 | 84,49 / 80,98 | 84,88 / 86,45 |

Chọn kích thước giải thích 78,3% và 89,7% mức tăng mAP so với một vùng ngẫu nhiên trên Ext-1 và Ext-2. Phần vị trí sau khi giữ nguyên kích thước là +0,294, +0,196 và −0,018 điểm mAP. So với vị trí cố định tốt của gallery ở cùng kích thước, chênh lệch là −0,170, −0,091 và −0,224 điểm. Kết quả chưa xác nhận lợi ích của chính sách chọn vị trí theo truy vấn. Có những lỗi thực sự phụ thuộc mạnh vào vị trí. Trong 91 trường hợp Ext-1 trả về ốc sên cho sketch cá đuối, truy vấn ở giữa phân bố chênh AP đạt 71,69% ở vùng được chọn và 97,94% ở vùng hậu nghiệm tốt nhất. Trên TU, với 60 trường hợp nhầm cup thành mug, con số tương ứng của truy vấn đại diện chỉ là 47,18% và 47,32%. [Phân rã paired, khoảng tin cậy và ví dụ thật](locked-error-diagnosis.json) · [Pilot](PILOT_VI.md) · [Giao thức khóa](full-protocol.json)

Chọn cấu hình trên hai split Sketchy unseen, dùng trung bình mAP rồi đến precision khi hòa. TU-Berlin dùng cấu hình đã khóa. Cả hai benchmark đều đo chiều sketch → photo. OpenAI CLIP ViT-B/32 được đóng băng. Hàm KD-SBIR gốc chấm trên đủ 14.179, 12.694 và 2.130 sketch cho mỗi cấu hình cuối. Ext-1 và TU dùng mAP@all và P@100. Ext-2 dùng mAP@200 và P@200. Mỗi lựa chọn được đối chiếu trên thứ hạng riêng, gồm đủ 13 hành động cho mỗi truy vấn. Hàng ngẫu nhiên lấy kỳ vọng của các metric gốc. Hàng hậu nghiệm lấy vùng cho AP cao nhất và dùng precision của chính vùng đó. [Giao thức](../../REGION_ACQUISITION_FULL.md) · [Khóa](freeze.json) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU](tuberlin.json) · [Từng truy vấn](per-query.zip)
