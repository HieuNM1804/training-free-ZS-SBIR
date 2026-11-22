# Nội suy từ ở đầu vào CLIP và kiểm tra trên các ảnh khác trong nhóm

Một nhóm ảnh đề xuất các tên Open Images. Các token cùng vị trí trong những tên có độ dài tương thích được lấy trung bình trước bộ mã hóa văn bản. Ảnh khác trong cùng nhóm quyết định mức tin cậy dành cho phép nội suy. Giả thuyết ban đầu chuẩn hóa theo toàn gallery. Lượt đầy đủ giữ giả thuyết này và kiểm tra thêm biến thể chỉ dùng bằng chứng nội bộ của nhóm, sau khi biến thể đó thắng trên pilot.

| mAP / precision (%) | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin 30 lớp |
|---|---:|---:|---:|
| Đối chứng vùng gốc | 84,00 / 83,45 | 76,43 / 74,17 | 79,50 / 80,13 |
| Nội suy với mức tin cậy chuẩn hóa theo gallery | 83,63 / 83,43 | 75,95 / 73,99 | 79,71 / 80,65 |
| Nội suy token với mức tin cậy trong nhóm | 83,94 / 83,61 | 76,54 / 74,36 | 79,55 / 80,41 |
| Nội suy đầu ra CLIP với đúng mức tin cậy trên | 83,26 / 83,22 | 75,74 / 73,94 | 79,12 / 79,87 |
| Đối chứng chọn vùng theo từng ảnh ứng viên | 84,37 / 83,85 | 77,01 / 74,84 | 80,07 / 80,92 |
| Thêm cùng bước nội suy token vào đối chứng chọn vùng | 84,34 / 84,04 | 77,12 / 74,97 | 79,99 / 80,68 |

Nội suy token tốt hơn nội suy embedding đầu ra với cùng hệ số trên cả ba split. So với điểm gốc trước khi thêm nội suy, thay đổi mAP là −0,057, +0,116 và +0,052 điểm. Cả ba khoảng tin cậy 95% bootstrap theo lớp đều chứa 0. Khi chuyển bước nội suy sang hệ thống chọn vùng theo từng ứng viên, thay đổi tương ứng là −0,030, +0,108 và −0,082 điểm. Bằng chứng hiện có xác nhận vị trí nội suy ảnh hưởng kết quả, nhưng chưa cho thấy cơ chế mới cải thiện ổn định so với bỏ nội suy. [Pilot gốc](PILOT_VI.md) · [Cấu hình follow-up](followup-protocol.json)

Chọn cấu hình trên hai split Sketchy unseen, dùng trung bình mAP rồi đến precision khi hòa. TU-Berlin dùng cấu hình đã khóa. Cả hai benchmark đều đo chiều sketch → photo. OpenAI CLIP ViT-B/32 được đóng băng. Hàm KD-SBIR gốc chấm trên đủ 14.179, 12.694 và 2.130 sketch cho mỗi cấu hình cuối. Ext-1 và TU dùng mAP@all và P@100. Ext-2 dùng mAP@200 và P@200. Biến thể dùng 8 tên mỗi nhóm, trọng số hiệu chỉnh 0,5 và hệ số 4 cho mức tin cậy. [Giao thức](../../LOCAL_TEXT_GATE_FOLLOWUP.md) · [Khóa](freeze.json) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU](tuberlin.json) · [Từng truy vấn](per-query.zip)
