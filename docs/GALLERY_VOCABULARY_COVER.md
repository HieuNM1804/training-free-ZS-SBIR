# Chọn từ vựng đại diện cho gallery bằng CLIP

Đây là tài liệu tái lập một hướng đã bị loại sau phép kiểm tra chuyển giao. Cấu hình thắng trên hai split Sketchy giảm P@100 từ 80,61% của mốc trước xuống 73,01% trên TU-Berlin. [Số liệu đã khóa và đối chứng](results/gallery-vocabulary-cover/summary.json)

Đối với các lớp dễ lẫn, nhiều sketch khác nhau có thể cùng nhận một danh sách ảnh sai rất nhất quán. Một giả thuyết là mỗi ảnh được mô tả bằng các từ đúng cục bộ, nhưng những từ đó nhấn mạnh chi tiết phụ hoặc bỏ sót một phần gallery. Phương pháp chọn một danh mục từ **chung cho gallery** trước khi xử lý truy vấn. Vật thể trong từng ảnh và các từ có ích đều được suy ra từ OpenAI CLIP ViT-B/32 đóng băng.

## Lập chỉ mục

1. Mã hóa 20.931 tên Open Images bằng prompt photo và sketch. Mỗi ảnh bỏ phiếu mềm cho 16 prompt photo gần nhất trong toàn bộ danh mục. Giữ lại 2.048 tên nhận tổng số phiếu lớn nhất làm ứng viên. Toàn bộ gallery tham gia bước này.
2. Lấy mẫu ngẫu nhiên cố định 1.536 ảnh gallery không nhãn. Với ảnh $i$, gọi $a_{iw}$ là cosine giữa ảnh và prompt photo của từ $w$. Đặt $u_{iw}=\exp((a_{iw}-\max_v a_{iv})/0{,}035)$, trong đó cực đại lấy trên 2.048 ứng viên. Từ đầu tiên làm cực đại tổng $u_{iw}$ trên các ảnh tham chiếu. Những lần tiếp theo thêm từ làm tăng nhiều nhất tổng mức độ được đại diện $F(S)=\sum_i\max_{w\in S}u_{iw}$. Một từ được nhiều ảnh ủng hộ vẫn có thể có lợi ích biên thấp nếu các ảnh đó đã được những từ khác đại diện tốt. Cấu hình đã khóa dừng ở **64 từ**.
3. Với mỗi ảnh trong toàn gallery, lấy softmax trên cosine của **cả 64 prompt photo**, nhiệt độ 0,025. Giữ trọng số, thay từng prompt photo bằng prompt sketch của cùng từ rồi chuẩn hóa tổng vector. Lưu vector này bên cạnh embedding RGB của ảnh đã trừ trung bình gallery và chuẩn hóa. Toàn bộ 64 từ được chọn bằng đặc trưng gallery, không cần biết số lượng hay danh tính các lớp benchmark.

Mục tiêu chọn tập con là dạng *facility location* với độ tương đồng không âm. Công trình về submodular subset selection dùng tập con đại diện cho dữ liệu. Ở đây, phần tử được chọn là một tên ngôn ngữ và phần tử cần được đại diện là một ảnh. FreeDom cung cấp tiền lệ đảo qua từ vựng rời rạc để chuyển miền. [Wei và cộng sự, ICML 2015](https://proceedings.mlr.press/v37/wei15.html) · [FreeDom, WACV 2025](https://arxiv.org/abs/2412.03297)

## Suy luận cho một sketch

Mã hóa nguyên sketch và so với toàn bộ vector biểu diễn bằng prompt sketch của gallery. Chuẩn hóa các điểm của riêng truy vấn này theo gallery. Lấy 256 ảnh đầu, chuẩn hóa trọng số của chúng bằng softmax nhiệt độ 0,6, rồi tạo vector trung bình có trọng số trong không gian RGB đã trừ tâm. Tính lại cosine giữa vector đó và gallery. Điểm cuối dùng 35% điểm ban đầu và 65% điểm phản hồi sau chuẩn hóa. Cấu hình đã chọn tắt phần lấy trung bình theo láng giềng ảnh khi lập chỉ mục. Bước phản hồi có tiền lệ trong truy hồi đối tượng. [Arandjelović và Zisserman, CVPR 2012](https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf)

## Các đối chứng

So 64 từ chọn theo độ bao phủ với 64 từ phổ biến nhất từ cùng tập ứng viên, 64 từ ngẫu nhiên từ toàn danh mục và mã đảo không hạn chế ở 64 từ. Giữ backbone, điểm phản hồi và evaluator như nhau. Các đối chứng bổ sung lần lượt bỏ phản hồi, thay prompt sketch bằng prompt photo, xáo danh tính từ giữa hai miền và chỉ dùng RGB. Phép thử độ ổn định thay seed lấy 1.536 ảnh tham chiếu sau khi khóa cấu hình và giữ nguyên tham số trên cả ba split. Nhãn lớp chỉ dùng sau truy hồi để chấm điểm hoặc giải thích sự trùng tên tự nhiên giữa từ vựng và benchmark.

Phiên bản chấm chính thức là `40211aa`. Phiên bản `42743ac` cho phép thiết lập seed của bước chọn tập tham chiếu để đo độ ổn định sau khi khóa. [Mã lập chỉ mục và xếp hạng](../tf_sbir/gallery_vocabulary_cover.py) · [Mã kiểm tra độ ổn định](../tf_sbir/lexicon_audit.py)
