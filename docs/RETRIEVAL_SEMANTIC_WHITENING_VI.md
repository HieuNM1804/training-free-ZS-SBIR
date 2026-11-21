# Whitening của mô tả ảnh bằng khái niệm công khai

Jégou và Chum phân tích tác dụng của trừ trung bình và whitening trên vector mô tả ảnh, đo riêng thông tin từ các đặc trưng đồng xuất hiện và tác dụng của giảm chiều. Chúng tôi kiểm tra liệu các từ liên hệ với nhau có làm điểm cosine giữa hai mô tả CLIP bị chi phối bởi một ít trục hay không. [Jégou và Chum, ECCV 2012](https://cmp.felk.cvut.cz/~chum/papers/Jegou-ECCV12.pdf)

Mỗi ảnh được đặt trên 20.931 từ công khai bằng softmax giữa embedding ảnh và hai prompt miền cố định. Lấy trung bình embedding CLIP của từng từ nguyên văn theo phân bố đó và chuẩn hóa. Trừ tâm, chiếu và đổi tỷ lệ các trục của mô tả này bằng ma trận hiệp phương sai của danh sách từ, rồi xếp hạng bằng cosine. Center và covariance dùng chung cho cả hai miền. Cấu hình định trước giữ 256 chiều, chia mỗi tọa độ cho căn bậc hai trị riêng. Dò 32, 64, 128, 256, 512 chiều và số mũ whitening 0, 0,5, 1 trên hai split Sketchy. Từ công khai chỉ dùng một lần để tính hệ trục. [Cách biểu diễn miền bằng từ, FreeDom, WACV 2025](https://arxiv.org/abs/2412.03297)

Nếu cải thiện đến từ whitening, phải có lợi so với cùng số chiều và chỉ trừ tâm. Nếu chỉ lợi khi ước lượng lại covariance trên ảnh của mỗi gallery, bằng chứng sẽ thiên về tác dụng của phân bố đích. Đối chứng còn có mô tả chưa xử lý, CLIP RGB với PCA gallery, xáo chỉ số từ ở một miền và likelihood nguyên từ.
