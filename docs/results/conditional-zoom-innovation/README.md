# Bằng chứng từ phần thông tin mới lộ ra khi phóng to

Với mỗi nhóm ảnh, ta hỏi liệu thay đổi phân bố từ của sketch từ độ phân giải gốc sang vùng phóng to có đi theo thay đổi mà chính các ảnh trong nhóm dự báo. Cả hệ thống gốc và mọi đối chứng dùng cùng CLIP, cùng các vùng của sketch và cùng gallery. Phép thử này đi tiếp từ phát hiện rằng quan sát gốc dự báo được từ xuất hiện khi zoom.

| mAP / precision (%) trên mẫu | Sketchy Ext-1 | Sketchy Ext-2 |
|---|---:|---:|
| Đối chứng chọn vùng theo từng ảnh ứng viên | 84,64 / 83,96 | 80,63 / 78,38 |
| Thêm điểm cho phần thông tin mới tiết lộ | 84,61 / 83,94 | 80,56 / 78,37 |
| Hoán vị ghép cặp quan sát gốc và zoom giữa các ảnh trong nhóm | 84,65 / 84,00 | 80,73 / 78,47 |
| Thay quan sát gốc bằng một vị trí khác trên sketch | 84,64 / 83,94 | 80,47 / 78,35 |

Trong 145 và 161 sketch có top-1 sai ở điểm gốc, tín hiệu đề xuất cho nhóm ảnh đúng lợi thế so với lớp đứng đầu sai ở 54,5% và 50,9% truy vấn. Đối chứng ghép sai cặp đạt 59,3% và 66,5%. Việc dự báo được từ sẽ xuất hiện ở độ phân giải cao chưa cung cấp bằng chứng đáng tin cho lớp của ảnh cần tìm. Biến thể chính giảm mAP trên cả hai mẫu và chưa được chuyển sang TU-Berlin.

Pilot chọn theo trung bình mAP của 900 và 756 sketch với toàn gallery. Cấu hình dùng hệ số kernel 20 và trọng số 0,25. Chênh lệch hai lần đọc bằng 0 cho điểm tương tác bằng 0 chính xác. Điểm trên mẫu được tính bằng bộ chấm vector hóa. [Công thức và tài liệu](../../CONDITIONAL_ZOOM_INNOVATION.md) · [Kết quả tất cả cấu hình](pilot.json) · [Từng truy vấn](pilot-per-query.zip)
