# Một cặp ảnh có dự đoán được chênh lệch trên các tên bị giữ lại?

Với từng sketch và ảnh ứng viên, CLIP chấm cùng các tên công khai bằng hai cách mô tả miền. Phương pháp dùng chênh lệch giữa các cách mô tả để giải thích vì sao phản hồi của hai ảnh khác nhau. Khi dự đoán một tên, phương pháp loại phản hồi của chính tên đó khỏi phép ước lượng. Kỳ vọng là một cặp ảnh đúng có thể dự đoán được cả những tên giữ lại. Công thức đóng cho cùng kết quả với việc thực sự bỏ từng tên và giải lại, với sai số số học dưới 5 × 10⁻¹⁵ trong phép kiểm tra. [Giao thức và tiền lệ tangent distance](../../PREDICTIVE_DEPICTION_DISTANCE.md)

| mAP / precision (%) trên mẫu 36 sketch mỗi lớp | Ext-1 | Ext-2 |
|---|---:|---:|
| Mốc CLIP dùng từ và vùng | 84,64 / 83,94 | 80,64 / 78,38 |
| Hiệu chỉnh theo sai số dự đoán tên giữ lại | 84,65 / 83,99 | 80,59 / 78,32 |
| Thay bằng hai hệ số cố định cho mọi cặp, chọn riêng trên Sketchy | 84,63 / 84,01 | 80,79 / 78,44 |
| Giữ lại cả tên và ba tên gần nghĩa để chấm | 84,66 / 83,99 | 80,60 / 78,32 |
| Giữ lại tên và ba tên ngẫu nhiên, cùng cấu hình | 84,66 / 83,99 | 80,60 / 78,32 |

Trong 200 ảnh đầu của mỗi sketch, chúng tôi xét các cặp kề nhau có một ảnh đúng và một ảnh sai. Trên Ext-1, khi ảnh sai đang đứng trên ảnh đúng, tín hiệu mới ưu tiên ảnh đúng ở 70,14% trường hợp, tính trung bình theo truy vấn có cặp hợp lệ. Nó thực sự đưa 515 trong 967 cặp đó về đúng thứ tự. Trên Ext-2, cùng phép đo chỉ đạt 47,81%, sửa 971 trong 2.688 cặp sai và làm đảo 1.093 trong 2.711 cặp đang đúng. Lợi thế ở mẫu Ext-1 không giải thích được thứ hạng trên Ext-2. Cả việc giữ lại tên gần nghĩa và việc ước lượng riêng cho từng cặp đều thiếu lợi ích so với đối chứng đơn giản.

Mẫu có 900 và 756 sketch, với gallery đầy đủ của từng split. Các điều kiện cuối đã được chấm qua hàm gốc KD-SBIR trên T4. Tham số được dò trên unseen. Nhánh dừng ở pilot và chưa được chấm TU-Berlin. [Kết quả pilot](pilot.json) · [Đối chứng hệ số cố định](held-name-controls.json) · [Phân tích từng loại đảo hạng](hard-boundary-audit.json) · [Điểm từng truy vấn](pilot-per-query.zip)
