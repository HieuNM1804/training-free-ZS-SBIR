# Cây từ vựng công khai cho truy hồi ảnh

Nistér và Stewénius dùng cây visual words với hierarchical k-means để truy hồi đối tượng. Một quan sát vào một nút lá cũng được đếm ở nút cha và được cân theo độ hiếm. Cách chấm của bài hỗ trợ khoảng cách L1 và L2 trên histogram chuẩn hóa. Chúng tôi chuyển phép lượng tử hóa sang toàn bộ 20.931 embedding của từ công khai, trước khi có ảnh benchmark. Mỗi từ là một tài liệu tham chiếu để tính độ hiếm của nó và các nút cha. [Vocabulary Tree, CVPR 2006](https://people.eecs.berkeley.edu/~yang/courses/cs294-6/papers/nister_stewenius_cvpr2006.pdf)

Trong bản chuyển sang SBIR, mỗi ảnh phân phối khối xác suất cho những từ mà CLIP đánh giá cao. Các nút cha thu gom khối lượng của các từ con, cho phép hai từ riêng biệt đóng góp vào cùng một nhóm. Cây có mười nhánh ở mỗi nút nếu đủ số phần tử và ba tầng phân nhóm. Tầng cuối có thể giữ nguyên từng từ làm lá. IDF của một nút là log của tổng số từ chia số từ đi qua nút đó. Khoảng cách L1 trên hai histogram chuẩn hóa tương đương với xếp giảm dần theo phần giao. Tính giao chính xác trên hỗ trợ thưa của query. [FreeDom, WACV 2025](https://arxiv.org/abs/2412.03297)

Tham chiếu định trước dùng 16 từ, đủ ba tầng và lá nguyên từ, IDF và giao histogram. Trên Sketchy thử 4, 16, 64 từ, có hoặc bỏ tầng nguyên từ, L1 hoặc cosine chuẩn hóa L2. So với chỉ tầng nguyên từ, IDF đều, một cây xáo quan hệ từ trong khi giữ cùng số node, và thay thứ tự xử lý đồng điểm. Nếu cây xáo vẫn giữ được kết quả, chưa thể quy lợi ích cho việc nhóm những từ gần nghĩa.

## Kết quả và giới hạn chuyển giao

Cây thu được 10, 100, 849 nút trong ba tầng nội bộ và có 20.931 lá từ. Hai split chọn 64 từ, dùng tất cả các tầng cùng IDF và giao histogram. Kết quả **58,55 / 61,40 / 58,85 mAP**. Đối chứng giữ nguyên 64 từ cùng cách tính nhưng chỉ dùng lá cho **71,09 / 67,22 / 71,07**. Đổi thứ tự gallery khi phá hòa gần như giữ nguyên điểm của cấu hình đã chọn. Thí nghiệm xác định việc cộng bằng chứng ở các nút cha làm mất tính phân biệt trong thiết lập này. Cấu trúc cây công khai dùng được ở cả ba split, còn lợi ích thuật toán cho SBIR chưa được xác nhận.

Các điểm Sketchy là kết quả thăm dò trên unseen. Tham số dùng chấm TU được khóa trên hai split Sketchy và hàm metric KD-SBIR được đối chiếu trên cùng thứ hạng của từng truy vấn. [Cấu hình và SHA của bản chạy](results/retrieval-vocabulary-tree/freeze.json) · [Bảng kết quả, ablation và phân tích theo lớp](https://github.com/HieuNM1804/training-free-ZS-SBIR/blob/research/map-retrieval-analysis/docs/IMAGE_RETRIEVAL_ROUND_RESULTS_VI.md)
