# Ghép tín hiệu ảnh và mô tả theo WeiMoCIR

Trong composed image retrieval, WeiMoCIR kết hợp ảnh truy vấn và câu mô tả thay đổi bằng tổng có trọng số, sau đó kết hợp độ tương đồng với ảnh đích và trung bình độ tương đồng với các caption của ảnh đích. Tác giả đánh giá trên FashionIQ và CIRR, dùng ba caption và báo hệ số α=0,1, β=0,8. Nghiên cứu dùng Gemini để tạo caption. Trong bản chuyển sang SBIR, chỉ CLIP B/32 sinh bằng chứng từ một từ điển công khai. [WeiMoCIR, TAAI 2024, phương trình 1–2](https://arxiv.org/pdf/2409.04918) · [Mã tác giả và thông tin hội nghị](https://github.com/whats2000/WeiMoCIR)

Tham chiếu giữ nguyên hai phương trình, ba mô tả và hai hệ số. Truy vấn là sketch cùng cụm “a photo.” chỉ miền đích. Ba mô tả của photo là ba khái niệm có logit cao nhất, đã mã hóa bằng prompt photo. Biến thể theo nội dung dùng các từ sketch kích hoạt rồi diễn đạt chúng bằng prompt miền photo. Các mô tả gallery cũng có thể được lấy trung bình theo toàn bộ phân bố từ của từng photo. Không sinh câu mới và không dùng thêm mô hình. Cách suy ra từ của ảnh có tiền lệ ở FreeDom. [FreeDom, WACV 2025](https://arxiv.org/abs/2412.03297)

Lỗi cần kiểm tra là một mô tả miền giống nhau cho mọi sketch có cung cấp đủ nội dung để ghép qua các miền hay không. Chấm riêng chỉ ảnh, chỉ mô tả, bỏ text truy vấn, dùng text miền cố định và xáo liên kết giữa photo với mô tả. Hệ số được chọn chung trên hai split Sketchy trước khi đo TU.

## Kết quả và giới hạn chuyển giao

Dùng mô tả phụ thuộc nội dung, α=0,1, β=1 và mọi từ của photo cho **65,97 / 68,52 / 65,45 mAP**. Giữ hệ số paper, text miền cố định và ba mô tả cho **56,25 / 60,96 / 56,22**. Với cùng cách biểu diễn gallery của ứng viên đã khóa, bỏ text truy vấn cho **67,84 / 68,75 / 66,52**. Phép thử cho thấy mô tả gallery hữu ích hơn việc làm phức tạp mô tả truy vấn ở đây. Không thể dùng kết quả này để đánh giá tác dụng caption của Gemini trong bài gốc. Phần sửa cho SBIR chỉ dùng CLIP và từ công khai.

Các điểm Sketchy là kết quả thăm dò trên unseen. Tham số dùng chấm TU được khóa trên hai split Sketchy và hàm metric KD-SBIR được đối chiếu trên cùng thứ hạng của từng truy vấn. [Cấu hình và SHA của bản chạy](results/retrieval-composed-modality/freeze.json) · [Bảng kết quả, ablation và phân tích theo lớp](https://github.com/HieuNM1804/training-free-ZS-SBIR/blob/research/map-retrieval-analysis/docs/IMAGE_RETRIEVAL_ROUND_RESULTS_VI.md)
