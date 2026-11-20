# Chẩn đoán thứ hạng sketch–ảnh của CLIP ViT-B/32

Hai lượt Colab đã lưu dùng cosine toàn ảnh đạt 41,64% P@100 trên Sketchy Ext-1 và 33,85% P@200 trên Ext-2. Thay đổi cả pipeline không xác định được vì sao ảnh sai đứng trên ảnh đúng. Phép thử dưới đây giữ nguyên gallery và thứ tự dữ liệu của KD-SBIR. Bản chạy mới dùng checkpoint OpenAI có SHA-256 bắt đầu bằng `40d365715913`.

## Đối chứng định trước

Lấy 24 sketch mỗi lớp bằng seed 1703 trên từng split Sketchy. Dùng nhãn để lấy mẫu chẩn đoán và đọc lỗi sau tính điểm. Đối chứng ảnh–ảnh lấy 24 ảnh khác trên mỗi lớp, loại chính ảnh truy vấn khỏi gallery. Kết quả trên mẫu cân bằng lớp dùng để chẩn đoán. Kết quả cuối cùng phải chấm mọi sketch bằng evaluator gốc của dự án.

| Giả thuyết | Phép đo và đối chứng có thể bác bỏ |
|---|---|
| Cùng một gallery, cosine giữa hai miền thị giác làm mất thông tin phân biệt lớp. | So top-1, P@K và hạng ảnh đúng đầu tiên giữa sketch–photo và photo–photo. Giữ nguyên sketch, lần lượt thử trừ tâm gallery, IsoCLIP, đọc khái niệm qua CLIP text, và đọc qua các cặp text bị hoán vị. Nếu xáo tên vẫn giữ mức tăng, giả thuyết về căn chỉnh ngữ nghĩa yếu đi. |
| Một số ảnh thu hút nhiều sketch sai vì vị trí đặc biệt trong không gian embedding. | Đếm tỷ phần các lần sai trong top 10 rơi vào 1% gallery được chọn sai nhiều nhất. Trừ cùng loại độ lệch NNN, dùng gallery bỏ đường chéo hoặc dùng prompt công khai làm ngân hàng đối chứng. Đếm số top-1 sửa đúng và số kết quả đúng bị phá. Việc lỗi tập trung tự nó chưa chứng minh chuẩn hóa sẽ giúp. |
| So sánh hai miền qua các khái niệm chung giữ được ngữ nghĩa mà cosine RGB bỏ sót. | Giữ cùng chỉ số từ, đối chiếu prompt nguồn với prompt đích và thử xáo liên kết giữa hai miền. Thay riêng biểu diễn đích từ ảnh thành mã từ vựng của ảnh. Lưu đường dẫn sketch, ảnh sai đầu tiên, ảnh đúng đầu tiên và hạng của nó để kiểm tra trực tiếp. |

## Nghiên cứu và phép đo chuyển giao

| Công trình | Hiện tượng và đối chứng của tác giả | Phần có thể kiểm tra trên ViT-B/32 |
|---|---|---|
| Cross the Gap, ICLR 2025 | Chuyển truy hồi ảnh–ảnh thành truy hồi liên modality cải thiện trên nhiều bộ dữ liệu. Thử chuyển phân lớp zero-shot theo chiều ngược lại để kiểm tra cách giải thích. | So cosine ảnh với phép đọc qua text. Dùng embedding của từ thật để tránh tối ưu pseudo-token bằng gradient. [Paper](https://arxiv.org/html/2502.04263v1) |
| IsoCLIP, CVPR 2026 | Khảo sát phần đẳng hướng và dị hướng trong projector ảnh–text và tác động khi lọc các phương phổ trên truy hồi cùng modality. | Chạy phép lọc 150/50 đã có trên đúng cùng ảnh, trước các bước đồ thị hoặc chuyển miền khác. [Paper](https://arxiv.org/abs/2603.19862) |
| Nearest Neighbor Normalization, EMNLP 2024 | Một số ứng viên nhận điểm cao cho nhiều truy vấn khác nhau. So điểm gốc với điểm trừ bias, kể cả khi thay ngân hàng tham chiếu bằng nguồn ngoài phân phối. | Ước lượng độ hút từ ảnh gallery hoặc từ các prompt cố định rồi kiểm tra thứ hạng của cùng một sketch. [Paper](https://aclanthology.org/2024.emnlp-main.1257/) |
| ASIF, NeurIPS 2023 | Cặp anchor cho phép kết nối hai encoder qua tọa độ tương đối. Tác giả khảo sát lượng dữ liệu ghép cặp và độ thưa của tọa độ. | Dùng cùng danh tính khái niệm trong hai template làm cặp. Kiểm tra đối chứng xáo danh tính và số láng giềng. [Paper](https://papers.neurips.cc/paper_files/paper/2023/hash/3186591903d9db31770ad131adb5ceb4-Abstract-Conference.html) |
| FreeDom, WACV 2025 | Ảnh lân cận cung cấp bằng chứng về nội dung truy vấn trước khi chuyển miền bằng text. Bài kiểm tra các thành phần của phép đảo qua ngôn ngữ. | So đối chứng phiếu tần suất đã chạy với phép liên kết mềm từng sketch và gallery qua các khái niệm đã duyệt. [Paper](https://openaccess.thecvf.com/content/WACV2025/html/Efthymiadis_Composed_Image_Retrieval_for_Training-Free_Domain_Conversion_WACV_2025_paper.html) |
| Local Self-Similarities, CVPR 2007 | Tương ứng hình ảnh còn có thể tồn tại trong quan hệ nội tại dù hai vùng khác hẳn về cường độ hoặc texture. Tác giả so với các cách đo tương đồng ảnh khác. | Đối chiếu nhánh patch đã chạy và kỹ thuật xáo tương ứng, làm giả thuyết cạnh tranh với sai lệch ở embedding toàn cục. [Paper](https://www.wisdom.weizmann.ac.il/~vision/SelfSimilarities.html) |
| FreeCP, ICCV 2025 | Một số từ có điểm kích hoạt nhưng bằng chứng không ổn định sau khi truyền qua quan hệ giữa các patch. Hai bước lọc dư thừa và khử nhập nhằng được đánh giá riêng. | Kiểm tra xem một tín hiệu text có còn phân biệt được ảnh đúng trong gallery hay chỉ phản ánh độ phổ biến chung. Đây là giả thuyết chuyển giao sang retrieval. [Paper](https://arxiv.org/html/2508.00557v1) |
| Efficient Diffusion on Region Manifolds, CVPR 2017 | Láng giềng của các vùng gallery có thể khôi phục đối tượng mà truy hồi trực tiếp bỏ sót. Tác giả đối chiếu đặc trưng toàn cục và theo vùng cùng các hình thức diffusion. | So điểm trực tiếp với phiên bản lan truyền trên gallery không nhãn. Có thể đo riêng đóng góp của điểm khởi đầu và cấu trúc lân cận. [Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Iscen_Efficient_Diffusion_on_CVPR_2017_paper.pdf) |

Các tài liệu trên là các giải thích cạnh tranh. Hướng chính sẽ được chọn theo lỗi quan sát được trên Sketchy. TU-Berlin chỉ được chấm sau khi ghi lựa chọn đó cùng cấu hình và checksum.
