# Chuyển cấu hình từ Sketchy sang đủ 30 lớp TU-Berlin

Hai nhánh training-free mới được chọn theo precision trên Sketchy. Khi mang nguyên cấu hình sang TU-Berlin, phương pháp dùng cặp prompt đạt **78,81% P@100 và 77,97% mAP@all**. Đối chứng mạnh nhất từ đợt Sketchy trước, có dùng contour và crop theo nét vẽ, đạt **82,51% và 80,01%**. Mỗi sketch truy hồi trong cùng gallery đủ 30 lớp. [Bảng từng lớp](results/tuberlin-per-class.csv) · [Số liệu từng cấu hình](results/sketchy-tu-scores.csv)

## Kết quả với một cấu hình cho cả ba cột

Mỗi ô ghi **precision / mAP**, đơn vị %. Ext-1 dùng P@100 và mAP@all, Ext-2 dùng P@200 và mAP@200, TU-Berlin dùng P@100 và mAP@all. Mọi dòng thực nghiệm của dự án dùng cùng checkpoint OpenAI CLIP ViT-B/32 đóng băng trên Colab L4. Mỗi truy vấn độc lập, chỉ dùng các view của chính nó và gallery không nhãn.

| Phương pháp | Từ điển | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin, 30 lớp |
| --- | --- | ---: | ---: | ---: |
| Cosine trực tiếp của CLIP | — | 41,64 / 30,29 | 33,85 / 36,34 | 46,27 / 29,06 |
| Đồ thị gallery, một view toàn ảnh | — | 60,78 / 64,67 | 52,60 / 52,94 | 69,02 / 68,42 |
| Đồ thị gallery, bốn view, đối chứng của nhánh patch | — | 61,41 / 64,46 | 52,92 / 53,27 | 70,91 / 70,03 |
| Tự tương đồng giữa patch CLIP | — | 68,62 / 63,72 | 59,85 / 61,58 | 73,12 / 58,43 |
| **Chuyển đặc trưng qua cặp prompt** | Open Images | **78,15 / 76,18** | **67,03 / 69,25** | **78,81 / 77,97** |
| Đối chứng dùng contour, chỉ ảnh | — | 82,99 / 82,58 | 78,03 / 79,40 | 81,58 / 77,80 |
| Đối chứng dùng contour và bộ chọn dựa trên gallery | Open Images | 83,64 / 81,93 | 78,93 / 80,92 | **82,51 / 80,01** |
| SeCo-SBIR, cùng ViT-B/32, có huấn luyện | Theo paper | 87,00 / 80,60 | 77,50 / 80,00 | 84,70 / 78,70 |

Đối chứng dùng contour còn cách SeCo-SBIR 2,19 điểm P@100 trên TU-Berlin. Nhánh cặp prompt còn cách 5,89 điểm P@100 và 0,73 điểm mAP@all. Đây là đối chiếu với số tác giả công bố. Kho Kaggle đang dùng có năm lớp ít hơn 400 ảnh gallery, trong khi paper mô tả ngưỡng ít nhất 400 ảnh mỗi lớp. Vì chưa có danh sách tệp của tác giả để đối sánh, các chênh lệch này có thể bao gồm khác biệt dữ liệu. [SeCo-SBIR, bảng 1 và phần thực nghiệm](https://arxiv.org/html/2608.03120v1#S4.T1) · [Số tệp từng lớp trong lượt chạy](results/tuberlin-kd-and-overlap.json)

## TU-Berlin kiểm tra được điều gì

- **Phương pháp dùng cặp prompt cải thiện precision trên cả 30/30 lớp TU-Berlin** so với cosine CLIP. Mức tăng chung là **32,54 điểm**, với khoảng bootstrap 95% theo lớp **[25,61, 40,03]**. Mức tăng nhỏ nhất là 5,54 điểm ở `wineglass`, lớn nhất là 76,37 điểm ở `house`. Cùng phương pháp tăng 36,51 điểm trên Sketchy Ext-1 và 33,19 điểm trên Ext-2. Đây là bằng chứng chuyển được giữa hai bộ dữ liệu trong cùng bài toán sketch–ảnh. Khả năng chuyển giữa những cặp miền khác vẫn là câu hỏi mở. [Kết quả ghép cặp, 5.000 lần bootstrap](results/sketchy-tu-summary.json) · [Bảng đủ 30 lớp](results/tuberlin-per-class.csv)
- **24 lớp có tên khác với các lớp unseen Sketchy.** Trên 1.700 sketch của nhóm này, nhánh cặp prompt đạt **79,07% P@100**, đối chứng contour có từ điển đạt **83,42%**, CLIP gốc đạt **47,87%**. Trên 430 sketch thuộc 6 lớp trùng tên, ba mức tương ứng là 77,80%, 78,90% và 39,93%. Mọi truy vấn đều xếp hạng toàn bộ 27.262 ảnh. Cách chia nhóm so sánh tên sau khi chuẩn hóa khoảng trắng, dấu gạch và chữ hoa. Nó vẫn có thể xếp các từ đồng nghĩa vào hai nhóm khác nhau. [Phân tích theo nhóm](results/sketchy-tu-summary.json)
- **Không có tệp trùng khớp tuyệt đối với các split Sketchy đang xét.** Đối chiếu 2.130 sketch TU với 24.642 sketch khác đường dẫn trong hợp hai split Sketchy, và 27.262 ảnh gallery TU với 27.447 ảnh Sketchy. Số khớp bằng 0 ở cả hash byte gốc và hash RGB sau resize 224×224. Hai phép kiểm tra này không phát hiện mọi biến thể gần trùng. [Đối chiếu tệp và nhãn](results/tuberlin-kd-and-overlap.json)

## Hai hướng nghiên cứu

**Chuyển đặc trưng qua các cặp prompt.** Mỗi khái niệm công khai được CLIP mã hóa theo mô tả miền sketch và miền ảnh. Với một ảnh gallery, phương pháp dùng 32 khái niệm láng giềng để xấp xỉ đặc trưng ảnh, rồi chuyển các hệ số ấy sang khái niệm tương ứng ở miền sketch. Khi truy hồi, mỗi view của truy vấn đọc cặp biểu diễn này tại 64 láng giềng gallery. Đồ thị ảnh tổng hợp phiếu của các view. Cấu hình thắng dùng sáu view, tỷ trọng ngữ nghĩa 0,35 và hệ số giữ phần dư ảnh khi tạo đại diện miền sketch bằng 0. Các crop cố định ở giữa khung và phép lật áp dụng được cho ảnh màu bất kỳ. Thuật toán cần mô tả hai miền và 20.931 khái niệm Open Images, không cần tên lớp benchmark. [Mã nguồn đúng commit](https://github.com/HieuNM1804/training-free-ZS-SBIR/tree/f06e76dc5412261706911fa4572dd5130188b5e0) · [Cấu hình đã khóa](results/sketchy-tu-prompt-lock.json)

Ba công trình cung cấp cơ chế chính là IsoCLIP cho không gian thị giác, Neighbor Embedding cho việc chuyển tọa độ giữa hai tập đại diện tương ứng, và Query Specific Fusion cho việc gộp thứ hạng theo lân cận. Giả thuyết nghiên cứu ở đây là dùng cặp mô tả miền để tạo cặp đại diện cho từng ảnh gallery ngay khi lập chỉ mục. Trong đối chứng cùng tham số ở cấu hình mặc định, bỏ đặc trưng miền của prompt làm giảm precision **4,48 và 3,24 điểm** trên hai split Sketchy. Xáo ngẫu nhiên ghép cặp với ảnh đích làm giảm **13,85 và 12,61 điểm**. Các đối chứng này ủng hộ vai trò của miền và đúng cặp ảnh trong cơ chế chuyển. [IsoCLIP, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Magistri_IsoCLIP_Decomposing_CLIP_Projectors_for_Efficient_Intra-modal_Alignment_CVPR_2026_paper.html) · [Neighbor Embedding, CVPR 2004](https://cse.hkust.edu.hk/~dyyeung/paper/pdf/yeung.cvpr2004.pdf) · [Query Specific Fusion, ECCV 2012](https://link.springer.com/chapter/10.1007/978-3-642-33709-3_47)

**Tự tương đồng giữa các patch CLIP.** Thay đổi màu sắc, chất liệu hoặc phong cách có thể làm các patch khác nhau về đặc trưng tuyệt đối mà vẫn giữ cách các vùng lặp lại trong cùng ảnh. Nhánh thuần ảnh dùng cùng phép biến đổi cho sketch và ảnh chụp, so quan hệ giữa 49 patch ở hai block CLIP rồi kết hợp với thứ hạng toàn ảnh. Cấu hình thắng trên cả hai split dùng cả hai block, 320 láng giềng gallery và tỷ trọng 0,10 cho phiếu quan hệ patch. Trên TU-Berlin, nó tăng **2,21 điểm P@100** so với đối chứng bốn view, nhưng mAP@all giảm **11,60 điểm**. So sánh này bao gồm thay đổi đồ thị và số láng giềng. Phép xáo vị trí ảnh trong ngân hàng đặc trưng, giữ nguyên các tham số khác trên Sketchy, cho thấy ghép đúng quan hệ patch đóng góp 1,31 và 0,79 điểm precision ở cấu hình kiểm tra. Ba nguồn là Self-Similarities, IsoCLIP và Query Specific Fusion. [Cơ chế tự tương đồng, CVPR 2007](https://www.wisdom.weizmann.ac.il/~vision/SelfSimilarities.html) · [Mã nguồn đúng commit](https://github.com/HieuNM1804/training-free-ZS-SBIR/tree/81116309a3d589772cca5458c52ed0c4625c5eb0) · [Cấu hình đã khóa](results/sketchy-tu-structure-lock.json)

## Dữ liệu, khóa tham số và tái lập

- Sketchy Ext-1 gồm 14.179 sketch, 17.100 ảnh, 25 lớp. Ext-2 gồm 12.694 sketch, 12.552 ảnh, 21 lớp. Mỗi split có 36 cấu hình của nhánh prompt và 23 của nhánh patch, gồm cả baseline và đối chứng. Việc tuning trên hai tập unseen này mang tính thăm dò. Mỗi split có trọng số một nửa khi chọn theo precision, mAP dùng để phá hòa. Phương pháp tổng quát được chọn chỉ từ các điểm Sketchy. Hai mốc dùng contour đã được chọn trên Sketchy trong lượt trước. [Kế hoạch phân tích](results/sketchy-tu-analysis-plan.json) · [Kết quả Sketchy trước đó](SKETCHY_COLAB_PAIRED_VI.md)
- Lúc khóa cả hai nhánh, ngày 20/11/2026 lúc 04:03:34 UTC, cả hai thư mục điểm TU đều có 0 kết quả. Lượt suy luận tiếp theo chỉ chấm bảy cấu hình đã định trước. Đồ thị trên bộ dữ liệu mới được xây từ chính gallery ảnh không nhãn của nó. [Bản ghi khóa chung](results/sketchy-tu-joint-lock.json)
- TU-Berlin dùng đủ **30/30 lớp unseen** theo đúng thứ tự của KD-SBIR. Bản Kaggle v1 có **2.130 sketch và 27.262 ảnh**, mỗi lớp có cả sketch và ảnh. Trên baseline, evaluator gốc đọc cùng thứ tự tệp, khớp từng bit với **4.260 giá trị theo truy vấn** và cả hai điểm trung bình. Kết quả toàn bộ phương pháp dùng cùng evaluator. [Kiểm tra với KD-SBIR](results/tuberlin-kd-and-overlap.json) · [Kho TU-Berlin đã dùng](https://www.kaggle.com/datasets/b20dccn616nguynhutun/tuberlin)
- Trong phiên Colab GPU, chạy ô cuối của `training-free.ipynb` hoặc hai lệnh dưới đây từ checkout chứa script. Chương trình lấy dữ liệu từ Kaggle, xác minh checkpoint, dùng hai Git worktree theo commit và lưu kết quả theo từng sketch. Lượt này dùng NVIDIA L4, 12 CPU, batch 64 và 2 worker cho ảnh toàn khung, batch 192 và 6 worker cho view crop. [Số đo thông lượng theo batch và worker](results/sketchy-colab-l4-paired-throughput.json)

```bash
python scripts/colab_crossdataset.py
python scripts/colab_crossdataset_report.py
```
