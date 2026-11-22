# Kiểm tra độ giống thị giác sau khi giữ cố định mô tả bằng từ

Tách các ảnh có phân bố từ gần nhau thành bốn phần. Hai phần dùng để ước lượng các hướng đồng biến giữa toàn ảnh và vùng ảnh, phần thứ ba đo lại tương quan, phần cuối cung cấp ảnh cần xếp hạng. Mỗi cặp sketch–photo được chấm theo khả năng chúng chia sẻ phần tín hiệu thị giác còn lại sau khi trừ trung bình của nhóm từ. Cùng một sketch được xử lý độc lập với mọi sketch khác.

| mAP / precision (%) | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin 30 lớp |
|---|---:|---:|---:|
| Mốc vùng và từ | 84,37 / 83,85 | 77,01 / 74,84 | 80,07 / 80,92 |
| Thêm cosine ảnh gốc, hệ số đã chọn riêng | 84,27 / 83,86 | 76,96 / 74,81 | 79,89 / 80,76 |
| Thêm bằng chứng thị giác có điều kiện theo nhóm từ | 84,25 / 83,86 | 76,93 / 74,75 | 79,70 / 80,81 |
| Mô hình tương tự nhưng chỉ trừ trung bình toàn gallery | 84,28 / 83,78 | 76,80 / 74,58 | 79,94 / 80,76 |
| Đảo ghép toàn ảnh và chi tiết trong từng nhóm từ | 84,29 / 83,79 | 76,73 / 74,83 | 79,89 / 80,72 |
| Giữ nguyên tổng khối lượng của mỗi nhóm từ khi cập nhật | 84,28 / 83,87 | 77,03 / 74,83 | 79,87 / 80,90 |

Cấu hình chính chọn 16 hướng, hệ số điều hòa 0,05 và hệ số ghép 0,04. So với mốc, mAP thay đổi −0,117, −0,082 và −0,375 điểm. Trong 1.531 sketch Ext-2 mà AP của cosine gốc cao hơn mốc ít nhất 5 điểm, mức tăng là +0,735 điểm. Trên các sketch Ext-2 còn lại, mAP giảm 0,194 điểm. Việc ghép đúng hai cách nhìn của cùng một ảnh trong gallery có thể xác minh được, nhưng độ giống đó chưa giúp sửa thứ hạng lớp ổn định giữa sketch và photo. Nhánh dừng ở đối chứng này.

Chọn cấu hình trên hai split Sketchy unseen, dùng trung bình mAP rồi đến precision khi hòa. TU-Berlin dùng cấu hình đã khóa. Cả hai benchmark đều đo chiều sketch → photo. OpenAI CLIP ViT-B/32 được đóng băng. Hàm KD-SBIR gốc chấm trên đủ 14.179, 12.694 và 2.130 sketch cho mỗi cấu hình cuối. Ext-1 và TU dùng mAP@all và P@100. Ext-2 dùng mAP@200 và P@200. Hai cách nhìn là toàn ảnh và trung bình chuẩn hóa của các vùng. Các lớp benchmark chỉ xuất hiện trong bước đánh giá. [Giao thức](../../WORD_CONDITIONAL_VISUAL_VERIFICATION.md) · [Khóa](freeze.json) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU](tuberlin.json) · [Từng truy vấn](per-query.zip)
