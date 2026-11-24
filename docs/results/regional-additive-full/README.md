# Kiểm định toàn bộ framework sau khi tách hiệu ứng kích thước và vị trí

Cùng một lần chạy trên Colab Tesla T4, seed 42, OpenAI CLIP ViT-B/32 đóng băng. Biến thể thay đúng điểm vùng bằng L10 + L01 − L00. Điểm ngữ nghĩa toàn ảnh, hệ số vùng 0,35, từ điển công khai 20.931 tên, GCWE và evaluator KD-SBIR giữ nguyên trong từng cặp. Các hệ số được phát triển trên Sketchy unseen. TU-Berlin gồm đủ 30 lớp, chấm sau khi khóa công thức.

| Cấu hình | Sketchy Ext-1 · mAP@all / P@100 | Sketchy Ext-2 · mAP@200 / P@200 | TU-Berlin · mAP@all / P@100 |
|---|---:|---:|---:|
| Vùng gốc + ngữ nghĩa toàn ảnh | 84,627 / 84,082 | 77,555 / 75,328 | 80,569 / 81,466 |
| Vùng cộng hai hiệu ứng + ngữ nghĩa toàn ảnh | 84,637 / 84,070 | 77,620 / 75,386 | 80,591 / 81,554 |
| Framework ba thành phần gốc | 84,783 / 84,249 | 77,860 / 75,672 | 81,495 / 82,185 |
| Framework ba thành phần, vùng cộng hai hiệu ứng | 84,810 / 84,277 | 77,914 / 75,752 | 81,475 / 82,243 |

Dòng cuối so với dòng trên tăng 0,028 / 0,054 mAP ở hai split Sketchy và giảm 0,020 mAP ở TU. Precision đổi +0,028 / +0,080 / +0,058 điểm. Cả ba khoảng bootstrap 95% cho thay đổi AP trung bình theo lớp đều chứa 0. Biến thể không đủ bằng chứng để thay baseline chính. Bỏ GCWE, chênh lệch mAP của hai bộ đọc là +0,010 / +0,065 / +0,022. Các con số rất gần nhau nên cần đối chiếu cùng lần chạy và cùng GPU.

### Ngoài các truy vấn dùng để phát hiện và xác nhận

Do một số sketch có mặt ở cả hai split, đã loại **hợp** các ID truy vấn từng xuất hiện trong lượt phát hiện hoặc xác nhận của biến thể này ở bất kỳ split nào. Với GCWE bật, mức tăng là +0,027 mAP trên 8.997 sketch Ext-1 và +0,025 trên 7.640 sketch Ext-2. Mức tăng precision tương ứng là +0,031 và +0,068. Đây là kiểm tra trên các ID chưa dùng cho hai bước này; các nghiên cứu lịch sử của dự án từng đánh giá toàn bộ Sketchy. So sánh TU đo chuyển giữa hai bộ dữ liệu trong cùng chiều sketch → photo.

### Cơ chế và giới hạn

Lượt thử đầu tiên giữ hiệu ứng kích thước và tương tác nhưng bỏ hiệu ứng chính của vị trí. Nó làm giảm mAP ở cả hai split. Phát hiện phụ dẫn tới biến thể hiện tại: giữ cả hai hiệu ứng chính và bỏ hạng tương tác. Biến thể vượt các đối chứng ở mẫu xác nhận 3.072 truy vấn mỗi split, nhưng hiệu quả sau cùng với GCWE nhỏ và không làm tăng mAP trên TU. Điều đã chứng minh là có thể can thiệp riêng vào cách ứng viên chọn kích thước và chọn vị trí khi chấm một bộ đặc trưng CLIP cố định. Ích lợi truy hồi của việc loại hẳn tương tác cần thêm bằng chứng.

[Phân rã toán học và prior art](mechanism-vi.md) · [Tiêu chí khóa trước TU](freeze.md) · [Điểm từng split và ngoài mẫu](paired-effects.json) · [Biểu đồ khoảng bootstrap theo lớp](paired-class-effects.pdf) · [Hướng dẫn tái lập](reproduce.md) · [Kết quả theo truy vấn](paired-T4.zip)
