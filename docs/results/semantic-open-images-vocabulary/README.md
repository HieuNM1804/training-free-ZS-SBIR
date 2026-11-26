# So sánh ba bộ từ trong S

Chạy full unseen trên Colab Tesla T4 với seed 42 và CLIP ViT-B/32 đóng băng. Tính lại softmax cho sketch và photo trên từng bộ từ rồi xếp hạng toàn gallery bằng tích vô hướng của S. CSV boxable có đủ 601 tên duy nhất. Bộ ngẫu nhiên cũng dùng 601 tên và được lấy một lần từ 20.931 tên. Hai bộ rút gọn có 26 tên chung, và được khóa trước khi chấm cả ba split. [Protocol](protocol.md) · [Danh sách đã khóa](freeze.json) · [CSV boxable chính thức](https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv)

Mỗi ô là mAP / precision (%). Ext-1 và TU-Berlin dùng mAP@all / P@100. Ext-2 dùng mAP@200 / P@200.

| Cấu hình S | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin Ext (30 lớp) |
|---|---:|---:|---:|
| Open Images đầy đủ (20.931 từ) | 76,22 / 80,02 | 72,45 / 71,33 | 74,13 / 78,28 |
| Open Images boxable (601 từ) | 75,48 / 77,10 | 72,57 / 71,08 | 74,80 / 75,16 |
| Ngẫu nhiên từ tập đầy đủ (601 từ) | 56,77 / 61,22 | 57,43 / 54,91 | 48,00 / 51,40 |

| Chênh lệch boxable so với tập đầy đủ (điểm %) | Ext-1 | Ext-2 | TU-Berlin |
|---|---:|---:|---:|
| ΔmAP | −0,74 | +0,13 | +0,68 |
| ΔPrecision | −2,92 | −0,25 | −3,12 |
| Khoảng bootstrap 95% của ΔAP trung bình theo lớp | [−8,03 đến +4,61] | [−5,38 đến +5,28] | [−4,78 đến +4,91] |

Bộ boxable giữ mAP trong vòng 0,74 điểm của bộ đầy đủ ở cả ba split, với precision thấp hơn trên cả ba. Kết quả khi ghép cặp theo lớp biến thiên giữa các lớp. Giữ 601 từ boxable cho mAP cao hơn mẫu ngẫu nhiên cùng kích thước 18,71 / 15,14 / 26,81 điểm theo thứ tự Ext-1, Ext-2, TU-Berlin. So sánh ngẫu nhiên dùng một mẫu cố định.

Chấm đủ 14.179 / 12.694 / 2.130 sketch trên gallery 17.100 / 12.552 / 27.262 ảnh. Mọi truy vấn đều dùng lại KD-SBIR gốc. Sai lệch trung bình tuyệt đối lớn nhất khi chạy lại S đầy đủ là 0,0070 điểm so với mốc đã lưu. Commit chạy `2f808a8`.

[Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU-Berlin](tuberlin.json) · [Kết quả từng truy vấn](paired-T4.zip)
