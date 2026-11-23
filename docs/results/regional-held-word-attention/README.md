# Kết quả pilot trên Colab L4, seed 42

Trên 1.536 truy vấn mỗi split, việc loại từ đang chấm khỏi phép chọn vùng làm mAP điểm vùng riêng thay đổi từ 81,033 lên 81,654 trên Ext-1 và từ 71,950 xuống 71,397 trên Ext-2. Với điểm ngữ nghĩa toàn ảnh đi kèm, kết quả thay đổi từ 84,235 / 83,477 sang 84,427 / 83,919 và từ 75,893 / 73,528 sang 75,242 / 72,960 (mAP / precision). Khi loại một từ khác có tần suất gallery gần nhất theo cặp, điểm vùng là 81,031 và 71,957, gần như giữ nguyên. Dấu trái ngược giữa hai split bác bỏ giả thuyết rằng loại tự xác nhận của từ là một cải tiến nhất quán. Không chọn theo kết quả TU.

Đã chấm bằng evaluator KD-SBIR gốc và gallery đầy đủ. Mẫu phát triển ở hai split Sketchy có thể chồng lặp về ảnh hoặc sketch. [Giả thuyết và tiêu chí chọn](protocol.md) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [Theo truy vấn](paired-L4.zip)
