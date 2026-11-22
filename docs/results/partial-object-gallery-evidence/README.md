# Nhiều chi tiết của một sketch có cần được cùng một ảnh giải thích?

Trên cùng 900 và 756 sketch chọn cân bằng theo lớp, ta so việc bắt buộc các vùng rời nhau phải được một ảnh ủng hộ với việc mỗi vùng có thể được một ảnh khác trong cùng nhóm thị giác ủng hộ. Cả hai cách dùng cùng số lần chạy CLIP, từ điển, ảnh gallery và xác suất chọn từng vùng. Nhánh một vùng là đối chứng đơn giản hơn. Mọi cấu hình trong bảng xếp hạng trên toàn bộ gallery.

| mAP / precision (%) | Sketchy Ext-1 | Sketchy Ext-2 |
|---|---:|---:|
| Điểm gốc từ vùng và toàn sketch | 84,36 / 84,16 | 80,15 / 77,96 |
| Cùng một ảnh giải thích nhiều vùng | 84,41 / 84,55 | 80,00 / 77,56 |
| Các ảnh khác nhau trong cùng nhóm được giải thích từng vùng | 84,41 / 84,55 | 80,00 / 77,55 |
| Chấm từng ảnh trực tiếp | 84,43 / 84,60 | 79,98 / 77,57 |
| Gộp các khả năng chỉ đọc một vùng | **84,47 / 84,51** | **80,20 / 77,67** |

Các phương pháp đều chọn 128 từ, nhiệt độ 0,5 và hệ số ghép 0,2 theo trung bình mAP hai split. Ở xác suất hiện diện chi tiết 0,2, ràng buộc cùng ảnh chỉ thay đổi mAP trung bình khoảng +0,0005 điểm so với cho phép ảnh khác nhau. Tăng xác suất lên 0,85 làm chênh lệch tăng lên +0,017 điểm, đồng thời mAP trung bình của cả hai giảm. Điều này phù hợp với giả thuyết các vùng sketch thường lặp tín hiệu, nhưng chưa chứng minh đó là nguyên nhân. Kết quả chưa ủng hộ phát triển ràng buộc cùng ảnh thành đóng góp chính.

Lượt thăm dò dùng OpenAI CLIP ViT-B/32 đóng băng trên Colab L4, 36 sketch mỗi lớp, 17.100 và 12.552 ảnh gallery. Ext-1 dùng mAP@all và P@100. Ext-2 dùng mAP@200 và P@200. Đây là số trên truy vấn mẫu với bộ chấm vector hóa. Nhánh dừng ở pilot, chưa chấm TU-Berlin hoặc làm lượt xác minh đầy đủ bằng KD-SBIR gốc. [Cơ chế và bài báo](../../PARTIAL_OBJECT_EVIDENCE.md) · [Bảng tất cả cấu hình](pilot.json) · [Số liệu từng truy vấn](pilot-per-query.zip)
