# Hiệu chỉnh từ công khai cho truy hồi từng cặp

## Giả thuyết và dự đoán trước khi chấm

Thay đổi một nửa từ điển đã làm đổi mAP trong thí nghiệm trước. Một nguyên nhân có thể kiểm tra là một số từ giống nhiều từ khác trong không gian CLIP. Nếu điều đó làm giảm tính đặc hiệu khi ghép ảnh, chuẩn hóa theo mức tương đồng của chính các prompt công khai có thể cải thiện thứ hạng. Smith và cộng sự phát hiện một số từ xuất hiện quá thường xuyên trong kết quả dịch từ và đề xuất inverted softmax tại ICLR 2017. Chúng tôi chuyển mẫu số của phép dịch sang hai cách diễn đạt cùng từ trong CLIP. Cơ chế từ công trình thứ hai là dùng từ công khai làm cầu nối xuyên miền. [Smith và cộng sự, ICLR 2017](https://arxiv.org/abs/1702.03859) · [FreeDom, WACV 2025](https://arxiv.org/abs/2412.03297)

Với mỗi từ \(w\), tính \(b_s(w)=\log\sum_v \exp(\beta\,t_s(w)^\top t_p(v))\) và \(b_p(w)=\log\sum_v \exp(\beta\,t_p(w)^\top t_s(v))\). Đối với mỗi ảnh thuộc miền \(d\), trừ \(b_d(w)\) khỏi logit ảnh–từ của CLIP. Chuẩn hóa khối ảnh theo tổng, khối truy vấn theo giá trị cực đại, rồi lấy tích vô hướng hai vector. Mỗi điểm chỉ dùng hai ảnh đang so, trọng số CLIP đóng băng và 20.931 từ Open Images. Việc tiền tính bảng từ độc lập với bộ dữ liệu.

Cấu hình công bố trước dùng \(\beta=\tau\) là hệ số logit của checkpoint, giữ cả cặp cùng từ trong tổng, hiệu chỉnh cả hai phía. Thăm dò sáu lựa chọn dùng chung cho Sketchy, gồm \(\beta/\tau\) bằng 0,25, 0,5 hoặc 1 và giữ hoặc bỏ cặp cùng từ. Đối chứng gồm chỉ hiệu chỉnh một phía, đồng đều, và hoán vị mức hiệu chỉnh giữa các từ. Nếu hoán vị vẫn cho lợi ích tương tự hoặc hiệu chỉnh mất điểm trên cả hai split, bằng chứng không ủng hộ giả thuyết.

Phương pháp cạnh tranh theo trung bình mAP trên hai Sketchy với likelihood đồng đều và WinCLIP cửa sổ, đều chấm từng cặp. Khóa thứ tự bằng hai split trước khi chấm TU-Berlin 30 lớp. Dùng hai mô tả miền cố định cho sketch và photo. Kiểm tra cùng cặp khi cắt gallery, đảo thứ tự và tách batch truy vấn. So sánh pilot 48 sketch/lớp với toàn bộ gallery, sau đó dùng evaluator KD-SBIR trên toàn bộ query.
