# Những quan sát nào thay đổi được lỗi nằm ngoài khả năng của mọi phép ghép vùng?

Trong những quyết định sai dù chọn quan hệ giữa hai phần có lợi đến đâu, thay riêng bằng chứng đọc từ sketch. Giữ nguyên ảnh ứng viên, phần nghĩa của ảnh và evaluator. Các hàng toàn sketch dùng một lần đọc trực tiếp và cùng một từ điển ảnh. Trong hàng toàn sketch và toàn từ vựng, tích của hai lần trùng từ bằng bình phương của điểm ngữ nghĩa trực tiếp, nên thứ hạng chính là thứ hạng của bộ đọc ngữ nghĩa trực tiếp.

| Cách đọc sketch | Lỗi đã bị chặn được sửa, Ext-1 / Ext-2 | Mọi cặp, số sửa / làm sai Ext-1 | Mọi cặp, số sửa / làm sai Ext-2 | Độ chính xác cân bằng Ext-1 / Ext-2 |
|---|---:|---:|---:|---:|
| Hai phần ảnh và hai phần từ vựng | 0 / 0 | 0 / 0 | 0 / 0 | 58,21% / 47,51% |
| Toàn ảnh, còn chia từ vựng | 76 / 145 | 89 / 72 | 162 / 93 | 57,95% / 54,28% |
| Hai phần ảnh, toàn từ vựng | 52 / 86 | 60 / 58 | 96 / 133 | 57,95% / 45,44% |
| Toàn ảnh và toàn từ vựng | **116 / 216** | 129 / 93 | 232 / 164 | **62,82% / 55,25%** |

Mẫu số của cột lỗi đã bị chặn là 226 và 458. Khôi phục cả hai quan sát làm được điều mà việc thay quan hệ giữa các tập vùng không thể làm trong mô hình trước. Tuy nhiên, cũng có quyết định mới bị đảo sai. Khoảng 95% theo lớp cho mức thay đổi cân bằng của hàng cuối là [−4,11, 20,12] / [−2,36, 20,85] điểm. Không có cơ sở quy toàn bộ chênh lệch cho tương tác giữa các vùng: khi tách sketch thành hai dải, phép đọc trước đó đã bỏ cả dải ngăn cách ở giữa. Các số là chẩn đoán trên cặp, không phải mAP toàn gallery. [Protocol](protocol.md) · [Chứng chỉ từng cặp](../semantic-dependence-limits/README.md) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [Điểm từng cặp](paired-T4.zip)
