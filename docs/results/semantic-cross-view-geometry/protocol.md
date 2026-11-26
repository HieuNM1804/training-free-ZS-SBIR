# Kiểm chứng hình học giữa hai cách nhìn của cùng một truy vấn

Mốc là S dùng CLIP ViT-B/32 đóng băng, 601 từ boxable, `a drawing of a {t}.` cho sketch và `a photo of the {t}.` cho photo. S lấy trung bình hai posterior từ của sketch gốc và lật ngang, rồi chấm tích vô hướng với posterior từ của từng photo. Phép thử hình học xếp lại 256 ứng viên đầu theo S và đo thứ hạng toàn gallery.

## Bằng chứng đã có

Trong lượt 20.931 từ, G đứng riêng cạnh S cải thiện mAP +0,63 / +1,07 / +1,84 điểm trên Ext-1, Ext-2 và TU-Berlin. Khoảng bootstrap 95% theo lớp lần lượt là [−0,21, 1,66], [−0,87, 3,29] và [0,65, 3,08]. G đo tương quan khoảng cách ở các cặp patch rồi dùng ảnh đã xác minh để cập nhật phân bố từ. Một phép tách đại số về sau cho thấy trên Ext-1, phần tương quan độ xa tâm có thể tái tạo gần như toàn bộ mAP của phép đo cũ. Phần quan hệ sau khi trừ độ xa tâm thắng đối chứng xáo trên Ext-2 và thua trên Ext-1. Tác động của bố cục cần được kiểm tra trong đúng cấu hình 601 từ. [Ablation S+G](../minimal-srg-patches/README.md) · [Tách độ xa tâm](https://github.com/HieuNM1804/training-free-ZS-SBIR/blob/621128d/docs/results/gcwe-relational-geometry/README.md)

## Cơ chế từ nghiên cứu trước

1. Philbin và cộng sự xếp lại shortlist của hệ truy hồi bằng từ trực quan nhờ kiểm chứng hình học của các khớp cục bộ. Bài toán của họ tìm lại cùng một vật thể hoặc địa điểm, nơi phép biến đổi hình học thường dễ xác định hơn giữa hai thể hiện cùng lớp. Ở đây dùng shortlist từ S và fit phép đồng dạng bằng phân bố khớp patch của CLIP. [Fast Spatial Matching, CVPR 2007](https://www.robots.ox.ac.uk/~vgg/publications/2007/Philbin07/)
2. Zhang và cộng sự chỉ ra các feature có thể nhầm hướng trái–phải và dùng các view biến đổi ở test time để chọn tư thế có khoảng cách matching thấp. Ở đây dùng hai view để kiểm tra liệu một phép đồng dạng từ một view dự đoán được khớp trong view còn lại sau phép đổi tọa độ đã biết. [Telling Left from Right, CVPR 2024, mục 4.1](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_Telling_Left_from_Right_Identifying_Geometry-Aware_Semantic_Correspondence_CVPR_2024_paper.pdf)

Các hướng liên quan gồm Hough voting trên patch của Hyperpixel Flow và FGW trên khoảng cách giữa các phần. Repo đã thử một dạng FGW rời rạc trên năm tâm vùng từ CLIP. Hướng PEARL căn chỉnh không gian query–key ngay trong attention để cải thiện định vị trong một ảnh. Đó là cơ chế cải thiện đầu đọc patch cho một phép thử khác. [Hyperpixel Flow, ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Min_Hyperpixel_Flow_Semantic_Correspondence_With_Multi-Layer_Neural_Features_ICCV_2019_paper.pdf) · [FGW, ICML 2019](https://proceedings.mlr.press/v97/titouan19a.html) · [PEARL, CVPR 2026](https://arxiv.org/html/2603.21528v1)

## Phép thử

Với từng cặp sketch–photo, đọc 49 patch CLIP mỗi ảnh. Lấy trung bình giữa softmax theo hàng và theo cột của ma trận cosine, dùng logit scale của checkpoint, rồi chia cho số patch để ra phân bố khớp có tổng bằng 1. Làm riêng cho sketch gốc và sketch lật. Dùng chỉ số lưới ngược chiều ngang để đưa các hàng của bản lật về hệ tọa độ gốc của sketch.

Gọi hai phân bố sau khi quy đổi là `A` và `B`. Tìm phép đồng dạng hai chiều `F_A` tối thiểu hóa sai số bình phương tọa độ photo dự đoán theo khớp của `A`. Nghiệm đóng dùng hiệp phương sai tọa độ có trọng số, cho phép tịnh tiến, tỉ lệ, xoay và phản chiếu. Đo phần sai số được giải thích trên `B` so với dự đoán hằng bằng tâm photo ước lượng từ `A`. Lặp `B → A` và lấy trung bình. Một bản so tương phản trừ điểm tính khi cố tình giữ hệ tọa độ bị lật. Hai bản chia sẻ toàn bộ cách đọc patch.

Ví dụ với một bộ khớp giả luôn lấy hai patch cùng chỉ số lưới, biến đổi sẽ bị đảo trái–phải giữa hai view sau khi quy đổi. Kiểm tra tổng hợp đòi hỏi điểm +1 khi chuyển được hình học, +1 với một phép đổi tỷ lệ và tịnh tiến cố định, −1 khi cả hai khớp chạy theo chỉ số lưới, và 0 khi hai vị trí độc lập.

## Đối chứng và chọn cấu hình

- **S** chấm lại từ lệnh mặc định trên cùng phiên. Mọi biến thể dùng đúng shortlist của S.
- **Appearance** dùng trung bình cosine có trọng số theo cùng hai ma trận khớp. **Distance** dùng tương quan khoảng cách của G trước đây trên cùng khớp.
- **Within view** fit và chấm trong chính view đó. **Wrong frame** fit chéo giữa hai view với hàng của bản lật giữ theo hệ sai. **Radial shuffle** hoán vị tọa độ photo trong từng nhóm cùng bình phương khoảng cách tới tâm và dùng đúng cùng hai ma trận khớp.
- Chuẩn hóa log điểm S và điểm bổ sung trên 256 ảnh. Sắp xếp các ứng viên theo tổng hai điểm, với hệ số hình học 0,125 / 0,25 / 0,5 / 1 / 2. Phép chấm đọc toàn gallery và dùng thứ tự từ S cho các ảnh ngoài nhóm 256.
- Quét tất cả cấu hình trên full unseen Sketchy Ext-1 và Ext-2. Chọn giữa hai bản chấm chéo theo trung bình hai mAP, rồi precision khi hòa. Khóa cấu hình và các đối chứng trước khi chấm TU-Berlin đủ 30 lớp. Báo tương phản theo lớp và chấm lại các hàng chính bằng hàm KD-SBIR gốc.

Giả thuyết được ủng hộ khi điểm giữ lại bố cục thật cải thiện S và vượt các đối chứng dùng cùng sự giống nhau về hình ảnh. Nếu chỉ vượt khi fit và chấm chung một view hoặc khi giữ hệ tọa độ sai, đó là bằng chứng ưu tiên cách giải thích dựa trên lưới. Phép thử này kiểm tra độ ổn định trước phản chiếu. Chuyển sang các bộ dữ liệu sketch → photo đo khả năng chuyển bộ dữ liệu trong cùng cặp miền. Biến dạng tư thế ngoài họ đồng dạng và độ thô của 49 patch là hai giới hạn của estimator.
