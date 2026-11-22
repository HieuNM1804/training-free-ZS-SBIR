# Chọn một vùng trước khi CLIP đọc vùng đó

Một sketch toàn cảnh đưa ra một số nhóm ảnh ứng viên. Từ hai nửa gallery ảnh không nhãn, ta ước lượng vùng nào có khả năng đổi đúng thứ tự của các nhóm đang cạnh tranh. Chọn vùng xong mới dùng nội dung vùng của sketch để xếp hạng ảnh. Tất cả hàng một vùng dưới đây dùng cùng bộ đọc CLIP và hệ số ghép 0,4.

| mAP / precision (%) | Sketchy Ext-1 | Sketchy Ext-2 |
|---|---:|---:|
| Chỉ đọc toàn sketch | 82,31 / 82,18 | 76,57 / 75,38 |
| Một vùng ngẫu nhiên, kỳ vọng chính xác | 81,96 / 81,81 | 76,49 / 74,43 |
| Một vùng cố định chọn từ gallery | 83,19 / 83,38 | 78,83 / 76,47 |
| Chọn bằng độ lợi thông tin Shannon | 83,00 / 82,61 | 77,73 / 75,31 |
| Ước lượng lỗi thứ tự trong cùng nửa gallery | 83,35 / 82,90 | 77,87 / 75,47 |
| Kiểm tra quyết định thứ tự bằng nửa gallery còn lại | **83,36 / 82,59** | **78,99 / 76,60** |
| Gộp toàn bộ 13 vùng | 84,23 / 83,90 | 80,27 / 77,67 |
| Một vùng tốt nhất cho mỗi sketch, chọn hậu nghiệm bằng nhãn | 88,96 / 89,06 | 86,34 / 82,67 |

Đây là 900 và 756 sketch mẫu, cùng toàn bộ gallery của hai split. Cấu hình chính chọn theo trung bình mAP, dùng 12 nhóm đối thủ và hệ số 2 cho điểm toàn cảnh. Trong từng sketch, độ lợi dự báo xếp đúng thứ tự lợi ích mAP của hai vùng với tỉ lệ 58,5% và 58,2% trên những cặp phân biệt được. Phần lớn hành động chọn vùng 5×5 patch. Mức tăng so với một vùng cố định chỉ 0,17 và 0,16 điểm mAP, vì vậy cần kiểm soát cả vị trí lẫn kích thước trước khi quy lợi ích cho việc thích nghi theo truy vấn. Số liệu gộp 13 vùng dùng ngân sách suy luận cao hơn. Oracle dùng nhãn chỉ để đo dư địa. [Giao thức](../../GALLERY_REGION_ACQUISITION.md) · [Cấu hình pilot](pilot.json) · [AP, precision và vùng chọn của từng sketch](pilot-per-query.zip)
