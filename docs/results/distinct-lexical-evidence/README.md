# Hai xác nhận từ vựng trên ảnh gốc và ảnh lật

Chỉ nhánh S, toàn bộ unseen Sketchy Ext-1, Ext-2 và TU-Berlin 30 lớp. Dùng CLIP ViT-B/32 đóng băng, Colab T4, seed 42 và evaluator KD-SBIR gốc cho từng truy vấn. Mã chạy ở `a4f1b60`. Điều kiện láng giềng thuận nghịch trong cả hai bộ prompt tạo 2.953 cặp từ rời nhau. [Công thức và đối chứng](protocol.md)

| mAP / precision (%) | Ext-1 | Ext-2 | TU-Berlin |
|---|---:|---:|---:|
| S gốc | 76,22 / 80,02 | 72,45 / 71,33 | 74,13 / 78,28 |
| Hai hướng nhân điểm | 75,92 / 79,68 | 72,09 / 71,01 | 73,85 / 77,95 |
| Hai hướng phải khớp hai từ khác nhau | 75,72 / 79,52 | 71,61 / 70,46 | 74,62 / 78,19 |
| Hai hướng phải khớp hai nhóm từ khác nhau | 75,56 / 79,35 | 71,29 / 70,20 | 74,60 / 78,21 |
| Nhóm từ tráo cặp | 75,72 / 79,52 | 71,61 / 70,46 | 74,62 / 78,19 |
| S với prior gallery | 81,48 / 82,48 | 74,85 / 73,80 | 77,30 / 77,83 |
| Hai hướng nhân điểm, có prior | 81,23 / 82,21 | 74,55 / 73,47 | 76,95 / 77,45 |
| Hai từ khác nhau, có prior | 74,99 / 78,97 | 67,55 / 64,20 | 72,42 / 74,53 |
| Hai nhóm khác nhau, có prior | 73,75 / 78,66 | 66,75 / 63,22 | 71,92 / 74,17 |
| Nhóm từ tráo cặp, có prior | 74,99 / 78,97 | 67,55 / 64,20 | 72,42 / 74,53 |

Dòng được khóa trước TU là hai từ khác nhau với posterior gốc. Nó thay đổi mAP so với S **−0,50 / −0,83 / +0,49 điểm**, với khoảng bootstrap theo lớp chứa 0 ở cả ba split. Ghép với prior có sẵn làm mất **6,49 / 7,30 / 4,88 điểm** so với S có cùng prior. Gộp cặp từ gần nhau rồi yêu cầu hai nhóm khác nhau thấp hơn đối chứng tráo cặp trên cả ba split, nên chưa có bằng chứng giữ cơ chế nhóm.

Trong phân tích thăm dò sau chấm, chỉ xét các lỗi hạng nhất của S rồi lấy một phần tư có điểm ảnh sai phụ thuộc mạnh nhất vào một từ. Trên chính các tập này (880, 1.044 và 133 sketch), cơ chế hai từ khác nhau với posterior gốc tăng **0,48 / 1,38 / 1,88 điểm AP** so với đối chứng nhân hai hướng. Nhóm truy vấn được nhận diện bằng nhãn sau chấm. Kết quả mô tả hiệu ứng ở một loại lỗi, và chưa cung cấp quy tắc nhận diện loại lỗi lúc suy luận.

[Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU-Berlin](tuberlin.json) · [Khóa](freeze.json) · [Điểm từng truy vấn](paired-T4.zip)
