# Dùng tần suất thắng trong gallery làm prior cho các từ

Bước lấy mẫu chỉ ước lượng tần suất của mỗi từ từ toàn gallery. Sau đó mỗi ảnh dùng lại likelihood CLIP đầy đủ với prior này. Một bản sao có cùng embedding ở cả hai miền giữ nguyên tổng prior, bằng chứng của ảnh và điểm truy hồi. Thiết kế có thể kiểm tra độc lập với độ chính xác trên từ điển tự nhiên. [Phương pháp](../../LEXICAL_REFERENCE_PROTOCOL.md)

| mAP / precision (%) | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin 30 lớp |
|---|---:|---:|---:|
| Baseline | 81,86 / 81,33 | 74,19 / 72,68 | 79,16 / 80,24 |
| Prior từ tần suất thắng tương quan | 82,91 / 81,84 | 74,54 / 73,06 | 78,54 / 78,49 |
| Prior từ trung bình posterior gốc | 83,01 / 82,07 | 74,27 / 72,90 | 79,01 / 79,02 |
| Prior từ từ có logit cao nhất | 82,94 / 81,84 | 74,64 / 73,12 | 78,50 / 78,38 |
| Prior cực đại likelihood (MLE) | 83,25 / 81,98 | 74,89 / 73,53 | 78,09 / 77,69 |

Khi bỏ một từ khỏi 1.024 cặp gần nhau, số top-1 đổi giảm từ 729 xuống 323 trên Ext-1, từ 1.062 xuống 788 trên Ext-2 và từ 350 xuống 286 trên TU. Chỉ số này đếm danh tính ảnh; hai ảnh cùng lớp vẫn có thể đổi chỗ. Đổi seed của prior làm đổi lần lượt 371, 449 và 138 top-1. Baseline và các điểm số ở bảng là số đo của chính lượt chạy này, bao gồm sai khác nhỏ của phép tính GPU ở các ảnh sát điểm.

Lợi ích mAP trên hai split Sketchy cũng xuất hiện với prior tất định đơn giản. Khi chuyển sang TU, cấu hình đóng băng giảm 0,62 điểm mAP và 1,75 điểm precision so với baseline. Đây là bằng chứng về độ nhạy của khớp từ với cách biểu diễn từ điển. Hình học tương quan chưa chứng minh ưu thế về độ chính xác.

Dùng ViT-B/32, word bank công khai và hàm KD-SBIR gốc trên Colab L4. Ext-1 và TU dùng mAP@all / P@100, Ext-2 dùng mAP@200 / P@200. [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU](tuberlin.json) · [Khóa](freeze.json) · [Từng truy vấn](per-query.zip)
