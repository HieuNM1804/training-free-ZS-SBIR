# Cùng một biểu diễn danh từ cho cả sketch và ảnh

Bỏ cả hai prompt mô tả miền trong bước khớp từ. Mỗi danh từ từ điển được đưa nguyên văn qua text encoder một lần và dùng lại ở cả hai phía. Giữ nguyên vòng đề cử qua cosine, 49 patch, ảnh trống của chính truy vấn, hai nửa gallery và tám lượt rút trong luật xếp hạng. Chấm trực tiếp và với diffusion RGB để tách thành phần khỏi bộ điểm vùng mạnh có prompt riêng. Việc thay prompt làm thay đổi điểm xuất phát, vì vậy mỗi mức tăng so với hàng gốc trong **chính** bảng này. Sketchy dùng cùng 768 truy vấn ngẫu nhiên mỗi split, TU dùng đầy đủ 2.130 truy vấn của cả 30 lớp.

| mAP / precision (%) | Sketchy Ext-1 pilot | Sketchy Ext-2 pilot | TU đầy đủ |
|---|---:|---:|---:|
| Danh từ trần, khớp trực tiếp | 74,67 / 79,46 | 70,43 / 69,59 | 71,79 / 76,38 |
| Cập nhật bằng bố cục | 74,87 / 79,20 | 71,34 / 70,44 | 73,77 / 77,73 |
| Xáo vị trí patch | 73,47 / 78,41 | 70,07 / 69,21 | 70,51 / 75,02 |
| Dùng cosine toàn ảnh để thay bố cục | 67,23 / 73,34 | 67,39 / 65,81 | 70,31 / 75,30 |
| Khớp danh từ trần và diffusion | 82,20 / 81,94 | 73,29 / 71,49 | 79,74 / 79,95 |
| Cập nhật bằng bố cục và diffusion | 81,87 / 81,29 | 73,37 / 72,36 | 80,55 / 80,64 |

Cập nhật từ làm tăng mAP trực tiếp ở cả ba lượt. Trong lượt đầy đủ trên TU, mAP trực tiếp đi từ 71,79 lên 73,77 và P@100 đi từ 76,38 lên 77,73. Ở Ext-1, khi đưa qua diffusion, mAP và precision của pilot giảm. Công thức có thể hoạt động với cùng danh từ ở hai đầu và không có tên miền trong văn bản. Việc chuyển sang cặp miền ảnh khác vẫn cần dữ liệu kiểm chứng. [Sketchy](bare-words-sketchy.json) · [TU](bare-words-tuberlin.json).
