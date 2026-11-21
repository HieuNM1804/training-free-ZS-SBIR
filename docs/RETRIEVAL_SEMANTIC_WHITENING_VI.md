# Whitening của mô tả ảnh bằng khái niệm công khai

Jégou và Chum phân tích tác dụng của trừ trung bình và whitening trên vector mô tả ảnh, đo riêng thông tin từ các đặc trưng đồng xuất hiện và tác dụng của giảm chiều. Chúng tôi kiểm tra liệu các từ liên hệ với nhau có làm điểm cosine giữa hai mô tả CLIP bị chi phối bởi một ít trục hay không. [Jégou và Chum, ECCV 2012](https://cmp.felk.cvut.cz/~chum/papers/Jegou-ECCV12.pdf)

Mỗi ảnh được đặt trên 20.931 từ công khai bằng softmax giữa embedding ảnh và hai prompt miền cố định. Lấy trung bình embedding CLIP của từng từ nguyên văn theo phân bố đó và chuẩn hóa. Trừ tâm, chiếu và đổi tỷ lệ các trục của mô tả này bằng ma trận hiệp phương sai của danh sách từ, rồi xếp hạng bằng cosine. Center và covariance dùng chung cho cả hai miền. Cấu hình định trước giữ 256 chiều, chia mỗi tọa độ cho căn bậc hai trị riêng. Dò 32, 64, 128, 256, 512 chiều và số mũ whitening 0, 0,5, 1 trên hai split Sketchy. Từ công khai chỉ dùng một lần để tính hệ trục. [Cách biểu diễn miền bằng từ, FreeDom, WACV 2025](https://arxiv.org/abs/2412.03297)

Nếu cải thiện đến từ whitening, phải có lợi so với cùng số chiều và chỉ trừ tâm. Nếu chỉ lợi khi ước lượng lại covariance trên ảnh của mỗi gallery, bằng chứng sẽ thiên về tác dụng của phân bố đích. Đối chứng còn có mô tả chưa xử lý, CLIP RGB với PCA gallery, xáo chỉ số từ ở một miền và likelihood nguyên từ.

## Kết quả và giới hạn chuyển giao

Hai split Sketchy chọn 128 chiều và whitening đầy đủ bằng hệ trục tạo từ chính từ điển. mAP đạt **62,01 / 60,18 / 62,77** trên Ext-1, Ext-2 và TU-Berlin. Chỉ chiếu 128 chiều với cùng tâm cho 54,39 / 57,57 / 53,65. Giữ nguyên vector trung bình chưa xử lý cho 42,15 / 54,31 / 46,60. Phép whitening sửa một phần biến dạng của vector trung bình, nhưng chấm trực tiếp mức đồng thuận qua từng từ vẫn cho 75,31 / 71,42 / 73,35. Cơ chế whitening đã có trong truy hồi ảnh. Phần chuyển ở đây là hệ trục công khai cố định cho hai miền. Bằng chứng chưa ủng hộ chọn cách nén này làm phương pháp chính.

Các điểm Sketchy là kết quả thăm dò trên unseen. Tham số dùng chấm TU được khóa trên hai split Sketchy và hàm metric KD-SBIR được đối chiếu trên cùng thứ hạng của từng truy vấn. [Cấu hình và SHA của bản chạy](results/retrieval-semantic-whitening/freeze.json) · [Bảng kết quả, ablation và phân tích theo lớp](https://github.com/HieuNM1804/training-free-ZS-SBIR/blob/research/map-retrieval-analysis/docs/IMAGE_RETRIEVAL_ROUND_RESULTS_VI.md)
