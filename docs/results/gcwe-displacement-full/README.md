# GCWE dùng độ tập trung của các phiếu dịch chuyển

Chấm trên Colab T4 với OpenAI CLIP ViT-B/32 đóng băng, seed 42, 20.931 từ Open Images, gallery đầy đủ, evaluator KD-SBIR và từng sketch độc lập. Hai thành phần còn lại là Candidate-Conditioned Regional Matching gốc và Cross-Domain Semantic Retrieval gốc. Công thức được khóa ở commit 95ae088. Ext-1 có 14.179 sketch và 17.100 ảnh, Ext-2 có 12.694 sketch và 12.552 ảnh, TU-Berlin có 2.130 sketch và 27.262 ảnh thuộc 30 lớp.

| Bộ kiểm tra hình học trong GCWE | Ext-1 mAP@all / P@100 | Ext-2 mAP@200 / P@200 | TU mAP@all / P@100 |
|---|---:|---:|---:|
| Tắt GCWE | 84,627 / 84,082 | 77,555 / 75,328 | 80,569 / 81,466 |
| GCWE gốc | 84,783 / 84,249 | 77,860 / 75,672 | 81,495 / 82,185 |
| **Tập trung theo độ dịch chuyển** | **85,086 / 84,600** | **78,406 / 76,170** | **81,870 / 82,981** |
| Xáo tọa độ patch ảnh | 84,942 / 84,404 | 78,021 / 75,688 | 81,553 / 82,134 |
| Tính độ tập trung trước khi gom theo độ dịch chuyển | 84,943 / 84,350 | 78,110 / 75,731 | 81,734 / 82,447 |
| Gom theo độ dịch chuyển với trọng số thô | 84,969 / 84,397 | 78,514 / 76,215 | 81,446 / 82,451 |

**So với baseline ba thành phần**, can thiệp tăng mAP 0,303 / 0,546 / 0,375 điểm và precision 0,351 / 0,498 / 0,796 điểm. So với tắt GCWE, mức tăng mAP là 0,459 / 0,851 / 1,300 điểm. Cả ba split đều ưu tiên tọa độ thật hơn tọa độ xáo, và ưu tiên gom cặp patch theo độ dịch chuyển hơn chỉ đo độ sắc của từng cặp. Trừ phân bố vị trí kỳ vọng có hiệu quả khác nhau theo split. Trên Ext-2, dùng trọng số thô đạt mAP cao hơn 0,108 điểm. Việc chọn công thức chính diễn ra trước khi chấm TU.

**Độ chắc chắn.** Bootstrap ghép cặp lấy mẫu lại các lớp 10.000 lần, với hai phương pháp luôn chấm các sketch giống nhau. Khoảng 95% cho chênh lệch mAP có trọng số theo số truy vấn là [−0,309, +0,862], [−0,536, +1,667] và [−0,933, +1,435]. Số lớp tăng AP trung bình là 17/25, 14/21 và 19/30. Đây là ba cải thiện về ước lượng điểm với độ bất định theo lớp còn đáng kể. TU kiểm tra chuyển bộ dữ liệu trong chiều sketch → photo.

**Truy vấn ngoài giai đoạn phát triển của biến thể.** Loại hợp tất cả ID từng tham gia phát hiện hoặc xác nhận thống kê này trong bất kỳ split Sketchy nào. Trên 10.690 và 9.258 truy vấn còn lại, mAP tăng 0,202 và 0,551 điểm, precision tăng 0,176 và 0,484 điểm. Các nghiên cứu trước của dự án đã đánh giá cả hai split unseen. Đây là phép kiểm tra có giới hạn đối với việc chọn riêng biến thể GCWE.

**Thứ hạng thay đổi ở đâu?** Số truy vấn chuyển từ không có sang có ít nhất một ảnh đúng ở cutoff precision chính thức là 237 / 253 / 46. Số truy vấn mất ảnh đúng cuối cùng ở cutoff đó là 166 / 191 / 32. Trên Ext-2, riêng các truy vấn giữ được ít nhất một ảnh đúng trước và sau can thiệp đóng góp +0,497 trong tổng mức tăng +0,546 mAP. Ở lượt xác nhận, xác suất ảnh cung cấp bằng chứng thuộc đúng lớp tăng rất ít trên Ext-1 và giảm nhẹ trên Ext-2. Cơ chế cải thiện thứ hạng vẫn cần giải thích sâu hơn. [Phân tích ảnh cung cấp bằng chứng và biên ngữ nghĩa](../gcwe-displacement-donor-audit/README.md)

[Cách tính từng bước và prior art](mechanism-vi.md) · [Thiết kế khóa trước khi chấm](protocol.md) · [Lượt xác nhận](../gcwe-displacement-concentration/README.md) · [Khoảng tin cậy và chuyển trạng thái theo từng split](paired-effects.json) · [Biểu đồ so sánh](displacement-comparisons.pdf) · [Kết quả theo truy vấn](paired-T4.zip)

### Từ nào thực sự làm đổi thứ hạng?

Mỗi split lấy một sketch có ảnh đúng mới lọt vào cutoff và một sketch mất ảnh đúng cuối cùng. Trong mỗi nhóm, chọn trung vị độ lớn thay đổi AP trong các trường hợp AP cùng chiều với thay đổi hit. Hai split Sketchy cũng loại mọi ID đã dùng để phát hiện hoặc xác nhận biến thể. So sánh một ảnh đúng với ảnh cạnh tranh đứng đầu ở cấu hình bất lợi. Khi ảnh đúng đầu tiên thay đổi danh tính, hạng của một ảnh cố định có thể khác hạng của ảnh đúng tốt nhất.

| Truy vấn | Hạng ảnh đúng đầu tiên, gốc → mới | Điều quan sát được |
|---|---:|---|
| Ext-1, bướm | 287 → 91 | Các từ chỉ bướm và từ *bougainvillea* cùng thu hẹp cách biệt với ảnh gấu bông giữa hoa. |
| Ext-1, lạc đà | 79 → 491 | *Camel* và *camelid* góp tổng cộng −0,728 vào thay đổi biên điểm so với ảnh hươu. |
| Ext-2, chim hót | 583 → 152 | *American goldfinch* đóng góp +0,559 vào thay đổi biên +0,715 so với ảnh hải âu. Ảnh hải âu còn đứng hạng 2. |
| Ext-2, hải âu | 5 → 527 | Ảnh dơi vượt lên hạng 1. *Gull dong* có đóng góp âm lớn nhất vào thay đổi biên của cặp này. |
| TU, xe tải | 414 → 1 | Ảnh TV rơi từ hạng 1 xuống 2.088. Các từ về TV, màn hình và xe tải cùng đóng góp vào việc đảo thứ tự. |
| TU, máy bay | 1 → 237 | Trực thăng lên hạng 1. *Helicopter*, *airplane* và *plane (tool)* đóng góp âm vào thay đổi biên. |

Điểm đóng góp được tính qua toán tử diffusion chuyển vị. Với mỗi ảnh, tách thay đổi xác suất thành tổng theo từng từ, rồi đổi đơn vị sang chênh lệch log điểm bằng hệ số log(a/b)/(a−b). Các đóng góp cộng lại tái tạo thay đổi biên của cặp ảnh với sai số lớn nhất 0,000005 trên sáu trường hợp. Dấu dương có thể phản ánh giảm bằng chứng cho ảnh cạnh tranh hoặc tăng bằng chứng cho ảnh đúng. Sáu ví dụ mô tả các quyết định cụ thể. Tần suất gây lỗi của các từ đa nghĩa trên toàn gallery vẫn là câu hỏi mở. [Số liệu Sketchy](word-cases-sketchy.json) · [Số liệu TU-Berlin](word-cases-tuberlin.json) · [Mã quy đóng góp](../../../tf_sbir/gcwe_displacement_examples.py)
