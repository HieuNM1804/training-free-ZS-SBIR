# Quy tắc làm việc cho training-free ZS-SBIR

## Hỏi trước khi vượt phạm vi

Khi một lựa chọn triển khai có thể vượt phạm vi đã thống nhất hoặc có điểm mơ hồ làm thay đổi giả thuyết, protocol, dữ liệu hay kết quả so sánh, hãy hỏi người dùng trong khung chat. Nêu rõ lựa chọn và lý do, rồi chờ người dùng trả lời trước khi viết code cho lựa chọn đó, khởi chạy thí nghiệm hoặc đưa thay đổi lên GitHub. Im lặng không phải là đồng ý.

Áp dụng quy tắc này trước khi đổi backbone hoặc checkpoint, dùng một mô hình bổ trợ khác, đưa thêm dữ liệu hay nhãn vào thuật toán, đổi split hoặc evaluator, chuyển sang suy luận transductive, bật gradient hoặc huấn luyện, hoặc thay đổi phạm vi công bố. Có thể tiếp tục công việc độc lập đã được duyệt trong lúc chờ. Các chi tiết triển khai thông thường trong phạm vi đã rõ có thể xử lý trực tiếp.

## Phạm vi hiện hành

- Dùng cùng một OpenAI CLIP ViT-B/32 đóng băng cho baseline và phương pháp. Mọi thí nghiệm chạy inference, không dùng gradient qua mô hình và không dùng DINO. Có thể đọc paper dùng backbone khác để tham khảo cơ chế.
- Xử lý từng sketch độc lập. Thống kê, prototype và đồ thị dùng gallery ảnh không nhãn. Việc đánh giá dùng nhãn của benchmark sau khi tính điểm truy hồi.
- Dùng các split unseen từ data config và cách tính mAP/P@K trong KD-SBIR. Repo KD-SBIR là tham chiếu cho loader, split và evaluator.
- Từ điển công khai Open Images đã được chấp thuận như một nhánh báo cáo riêng. Không dùng danh sách lớp test Sketchy để chọn từ.
- Ghi rõ các cấu hình được chọn theo điểm trên unseen là kết quả thăm dò.
