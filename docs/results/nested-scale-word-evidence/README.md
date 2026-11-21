# Đối chiếu từ ở vùng lớn và vùng nhỏ trên cả sketch và photo

Các cửa sổ thực sự chứa nhau tạo một phân bố chung trên cặp từ. Đối chứng ghép độc lập giữ nguyên các phân bố biên ở từng kích thước. Phương pháp chính dùng bốn phép so sánh để tách phần phụ thuộc đồng thời vào cặp vùng trong cả sketch và ảnh gallery.

| mAP / precision (%) | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin 30 lớp |
|---|---:|---:|---:|
| Tương tác giữa hai miền sau khi giữ các biên | 83,98 / 83,52 | 76,27 / 74,05 | 79,39 / 80,27 |
| Hai kích thước độc lập, tối ưu trên Sketchy | 83,97 / 83,51 | 76,26 / 74,04 | 79,39 / 80,25 |
| Dùng trực tiếp cặp từ đã ghép | 84,02 / 83,58 | 76,33 / 74,11 | 79,43 / 80,36 |
| Chỉ phóng to các vùng sketch | 84,00 / 83,45 | 76,43 / 74,17 | 79,50 / 80,13 |

Đối với Ext-1, Ext-2 và TU, phần chênh mAP so với cùng phân bố biên xấp xỉ +0,01, +0,01 và dưới 0,01 điểm. Ghép trực tiếp thực tế cho mAP cao hơn phần tương tác đã tách riêng trên cả ba split. Dữ liệu hiện có không hỗ trợ lấy quan hệ chứa nhau làm đóng góp thực nghiệm chính.

Chọn cấu hình trên hai split Sketchy unseen, dùng trung bình mAP rồi đến precision khi hòa. TU-Berlin dùng cấu hình đã khóa. Cả hai benchmark đều đo chiều sketch → photo. OpenAI CLIP ViT-B/32 được đóng băng. Hàm KD-SBIR gốc chấm trên đủ 14.179, 12.694 và 2.130 sketch cho mỗi cấu hình cuối. Ext-1 và TU dùng mAP@all và P@100. Ext-2 dùng mAP@200 và P@200. [Giao thức](../../NESTED_SCALE_WORD_PROTOCOL.md) · [Khóa](freeze.json) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU](tuberlin.json) · [Từng truy vấn](per-query.zip)
