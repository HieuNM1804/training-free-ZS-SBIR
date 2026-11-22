# Cùng một sketch và cùng hai ảnh, việc đổi gallery làm đảo thứ tự bao nhiêu lần?

Hai split Sketchy có 2.231 sketch và 2.205 ảnh trùng đúng đường dẫn, thuộc bốn lớp kéo, cây, xe lăn và cối xay gió. Chúng tôi xây hai chỉ mục trên toàn bộ gallery gốc của từng split. Sau đó, với từng sketch chung, giữ đúng một ảnh cùng lớp và một ảnh khác lớp, duyệt hết các cặp như vậy rồi đo xem ảnh nào đứng trước ở mỗi chỉ mục. Tỷ lệ bên dưới được tính riêng từng sketch rồi lấy trung bình. Đây là phép đo phụ về sự phụ thuộc vào gallery. Bảng điểm toàn split vẫn dùng gallery và evaluator KD-SBIR gốc. [Giao thức](../../SHARED_GALLERY_DIAGNOSIS.md)

| Cách tính điểm trên các ảnh chung | Cặp đảo thứ tự | Cặp xếp đúng khi dùng gallery Ext-1 / Ext-2 | Mức trùng của 100 ảnh đầu |
|---|---:|---:|---:|
| Cosine CLIP gốc | 0,00% | 68,99 / 68,99% | 100,00% |
| Khớp trực tiếp toàn bộ tên, đọc cả sketch | 0,00% | 94,83 / 94,83% | 100,00% |
| Khớp từ toàn cục và diffusion RGB | 0,61% | 95,36 / 95,32% | 90,75% |
| Chỉ chấm vùng với chỉ mục 64 từ | 0,82% | 97,08 / 97,10% | 86,42% |
| Mốc mạnh kết hợp toàn cục và vùng | 0,60% | 96,96 / 96,88% | 89,65% |

Hai chỉ mục 64 từ chỉ có 10 tên chung. Dù vậy, tỷ lệ xếp đúng của bộ đọc vùng cao hơn bộ đọc trực tiếp toàn bộ tên ở cả hai gallery. Vì thế một phép sửa nhằm giảm độ nhạy của thứ hạng còn phải giữ được độ chính xác. [Kết quả và từng lớp](shared-scene-diagnosis.json) · [Từng sketch](shared-scene-diagnosis.zip)

## Tách việc chọn vùng và đọc vùng

Chia ngẫu nhiên ảnh gallery không nhãn thành sáu nhóm. Mỗi nhóm tự ước lượng bộ tên phù hợp với gallery và đặc trưng ảnh đại diện cho mỗi tên. Khi so một sketch với một ứng viên, nhóm thứ nhất chọn trọng số cho các vùng của sketch. Nhóm thứ hai dùng cách diễn giải riêng để đo ứng viên được các vùng đó ủng hộ đến mức nào. Các nhóm cùng chấm toàn bộ ảnh ứng viên và dùng chung đồ thị RGB. Phép thử so sánh điểm đọc khác nhóm với điểm đọc cùng nhóm. Cũng có đối chứng hoán vị vị trí vùng, đảo dấu hiệu chỉnh và chỉ lấy trung bình. [Giao thức, công thức và ba tiền lệ nghiên cứu](../../INDEPENDENT_GALLERY_REGION_READOUT.md)

| mAP / precision (%) trên mẫu 36 sketch mỗi lớp | Sketchy Ext-1 | Sketchy Ext-2 |
|---|---:|---:|
| Mốc từ và vùng | 84,64 / 83,96 | 80,63 / 78,38 |
| Giữ mốc rồi hiệu chỉnh bằng tỷ số điểm khác nhóm / cùng nhóm | 84,63 / 84,39 | 81,09 / 78,57 |
| Thay điểm vùng bằng kết quả chấm khác nhóm, chọn hệ số riêng | 84,67 / 83,94 | 80,56 / 78,34 |
| Lấy trung bình các phép đọc cùng nhóm, chọn hệ số riêng | 84,67 / 83,95 | 80,55 / 78,33 |

Cấu hình pilot chọn nhóm ảnh ngẫu nhiên, 32 từ mỗi nhóm, nhiệt độ mũ 1 và hệ số 3 cho log tỷ số. Chênh lệch mAP so với nền trên hai split là −0,006 và +0,462 điểm với khoảng bootstrap 95% theo lớp [−0,61, +0,43] và [−0,21, +1,11]. Pilot dùng 900 và 756 sketch trên gallery đầy đủ. Việc dò tham số trên unseen mang tính thăm dò. [Kết quả pilot và kiểm tra với KD-SBIR](pilot.json) · [Từng truy vấn](pilot-per-query.zip)
