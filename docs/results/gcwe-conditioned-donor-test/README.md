# Kết quả pilot trên Colab L4, seed 42

Trên 768 truy vấn mỗi split, GCWE gốc đạt 85,247 và 78,661 mAP. Khi chuẩn hóa theo các cặp kề nhau trong thứ hạng cosine, kết quả là 85,612 và 78,923. Ghép cặp ngẫu nhiên cũng đạt 85,662 và 78,619. Hiệu ứng matched trừ random lần lượt là −0,050 và +0,305 điểm mAP. Cả hai khoảng bootstrap theo lớp chứa 0. Biến thể không đạt tiêu chí chọn đã khóa. Trên chính mẫu này, chỉ bỏ GCWE đã đạt 85,540 và 78,796.

Khi thay bộ chọn vùng riêng của mỗi ứng viên bằng attention trung bình của gallery cho cùng sketch, mAP giảm 0,806 và 0,554 điểm. Bỏ hạng tử ngữ nghĩa toàn ảnh (giữ từ điển nội tại của hai nhánh còn lại) giảm 1,461 và 3,362 điểm. Hoán đổi định danh từ giữa các cặp có tần suất gallery gần nhau làm mAP truy hồi ngữ nghĩa có diffusion giảm từ 83,042 / 76,542 xuống 6,150 / 14,435. Điều đó cho thấy ánh xạ đúng danh từ đang mang thông tin quan trọng ở cả hai split; riêng thí nghiệm hoán đổi từ không thiết lập novelty.

Đã chấm bằng evaluator KD-SBIR gốc và gallery đầy đủ. Mẫu phát triển ở hai split Sketchy có thể chồng lặp về ảnh hoặc sketch. [Giả thuyết và tiêu chí chọn](protocol.md) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [Theo truy vấn](paired-L4.zip)
