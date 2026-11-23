# Kết quả pilot trên Colab L4, seed 42

Trên 1.536 truy vấn mỗi split, GCWE gốc đạt 84,325 / 83,462 trên Ext-1 và 76,720 / 74,452 trên Ext-2 (mAP / precision). Phép lấy kỳ vọng trên tập donor ngẫu nhiên với mỗi ảnh tự loại mình đạt 84,257 / 83,408 và 76,753 / 74,363. Thay đổi mAP trung bình hai split khoảng −0,018 điểm. Bật cho phép mỗi ảnh có mặt trong donor của chính nó chỉ đổi mAP tương ứng khoảng −0,001 và +0,002 điểm. Cách lấy kỳ vọng bỏ được sự phụ thuộc vào một phép chia donor, nhưng thí nghiệm không cho thấy cải thiện hiệu quả truy hồi. Trên cùng mẫu, bỏ toàn bộ GCWE đạt 84,235 / 83,477 và 75,893 / 73,528. Giữ baseline gốc và dừng chuyển biến thể này sang TU.

Đã chấm bằng evaluator KD-SBIR gốc và gallery đầy đủ. Mẫu phát triển ở hai split Sketchy có thể chồng lặp về ảnh hoặc sketch. [Giả thuyết và tiêu chí chọn](protocol.md) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [Theo truy vấn](paired-L4.zip)
