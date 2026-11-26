# Quyết định dừng thí nghiệm chia sẻ khái niệm giữa các vùng

Phương pháp đọc toàn sketch và bốn cửa sổ ở bốn góc, rồi tính xác suất các quan sát đó theo phân bố từ của từng ảnh gallery. Có 52 cách chia năm quan sát thành những nhóm cùng dùng một lần rút khái niệm. Mô hình tính tổng trên cả 52 cách. Phép thử đặt trước hai tiêu chí gồm lợi thế so với đối chứng trên cùng crop và mức tăng ít nhất 4 điểm mAP tuyệt đối trên mỗi split Sketchy. Cả hai split đều cho mAP thấp hơn baseline. [Giả thuyết, công thức, prior art và đối chứng](../../exchangeable-concept-observations.md) · [Giao thức đã khóa](protocol.json)

| Phương pháp, mAP / precision (%) | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin |
|---|---:|---:|---:|
| Dot 601 từ, toàn sketch | 75,35 / 77,10 | 72,66 / 71,07 | 73,46 / 73,98 |
| Chia sẻ khái niệm giữa một số vùng | 74,73 / 76,68 | 71,50 / 70,31 | 71,53 / 73,02 |
| Trung bình điểm từ cùng năm quan sát | 75,68 / 78,02 | 73,42 / 71,71 | 72,86 / 74,63 |
| Các vùng rút khái niệm độc lập | 75,01 / 77,02 | 72,15 / 70,77 | 72,00 / 73,44 |
| Cả năm dùng một lần rút | 67,90 / 70,32 | 63,10 / 61,83 | 65,63 / 67,29 |
| Hai trường hợp cực hạn với xác suất chia sẻ khớp theo cặp | 71,50 / 74,36 | 68,66 / 67,28 | 68,19 / 70,87 |

Mức thay đổi mAP so với Dot là −0,62, −1,17 và −1,93 điểm. Mô hình phân hoạch cao hơn hỗn hợp của hai trường hợp cực hạn trên cả ba split. Tuy nhiên, trung bình các quan sát vẫn cao hơn mô hình phân hoạch 0,96, 1,92 và 1,33 điểm mAP. Trên Ext-2, chuyển từ trung bình sang mô hình phân hoạch sửa được 209 kết quả top-1 và làm sai thêm 385 kết quả. Mức chênh AP trung bình theo lớp là −1,94 điểm với khoảng bootstrap 95% [−2,72; −1,13]. [Kết quả Ext-2](sketchy_2.json)

Phép đo trên các cặp có ảnh nguồn đã biết dùng tên của toàn sketch làm quan sát giữ lại sau khi đọc bốn vùng. Ảnh đối chứng có cùng lớp và điểm Dot gần ảnh nguồn nhất. Phương pháp cho lợi thế log dự báo trung bình của ảnh nguồn so với ảnh đối chứng chỉ 0,00161 nat trên Ext-1 và 0,00357 nat trên Ext-2. Các số này mô tả phép đo bổ sung ở mức ảnh gốc. Chỉ tiêu chính vẫn là truy hồi theo lớp. [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json)

## Phạm vi và tái lập

Chấm toàn bộ 14.179 sketch với 17.100 ảnh trên Ext-1, 12.694 với 12.552 trên Ext-2, và 2.130 với 27.262 trên TU-Berlin đủ 30 lớp. Ext-1 và TU dùng mAP@all / P@100. Ext-2 dùng mAP@200 / P@200. Bốn GPU Kaggle L4 chia việc mã hóa và dò tham số. Từng thứ hạng chính thức được phát lại qua evaluator KD-SBIR gốc. Mã thí nghiệm ở commit `7b42c89`, seed 42, OpenAI CLIP ViT-B/32 đóng băng. [Mã thực nghiệm](../../../tf_sbir/exchangeable_study.py) · [TU-Berlin](tuberlin.json)

Hai split Sketchy unseen được dùng để chọn tham số thăm dò với trọng số bằng nhau. Cấu hình thắng dùng cửa sổ rộng và cao bằng 75% ảnh cùng nồng độ Dirichlet 32, tương ứng xác suất hai vùng chia sẻ một lần rút là 1/33. Cấu hình này được khóa trước khi chấm TU-Berlin. Chi phí mã hóa một truy vấn ở cấu hình đã chọn là 10 lượt qua encoder ảnh gồm cả bản lật, so với 2 lượt của Dot. Nhãn và cặp ảnh nguồn chỉ được dùng trong đánh giá. [Cấu hình đã khóa](freeze.json) · [Số liệu từng truy vấn và thứ tự ID](paired-L4.zip)

Hai split Sketchy và TU-Berlin đều đo sketch → photo. Kết quả hiện tại chưa cung cấp bằng chứng để đưa mô hình chia sẻ khái niệm vùng thành đóng góp khoa học độc lập cho truy hồi xuyên miền. Đợt nghiên cứu được tạm dừng theo tiêu chí đã thống nhất.
