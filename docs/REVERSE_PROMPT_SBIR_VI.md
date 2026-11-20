# Vì sao nên chuyển biểu diễn ảnh gallery sang miền sketch?

Cùng một OpenAI CLIP ViT-B/32, chuyển từng ảnh gallery sang biểu diễn bằng prompt sketch rồi đối chiếu với sketch gốc đạt **80,14% P@100** trên Sketchy Ext-1, **71,59% P@200** trên Ext-2 và **80,61% P@100** trên đủ 30 lớp TU-Berlin. Cả ba số thuộc cùng một cấu hình. Đổi chiều chuyển miền và giữ nguyên các bước sau làm giảm precision trên cả ba split. Đây là bằng chứng chính để chọn phương pháp.

## Lỗi quan sát được

Trên cùng tập ảnh, CLIP phân biệt lớp tốt hơn nhiều khi truy vấn cũng là ảnh. Với 24 truy vấn mỗi lớp, precision của sketch so với ảnh là **41,05% so với 93,34%** ở Ext-1 và **34,30% so với 86,28%** ở Ext-2. Mỗi truy vấn ảnh đã được loại khỏi chính gallery của nó. Trên toàn bộ Ext-2, sketch tòa nhà `n04233124_9355-1.png` có ảnh chiếc kéo `scissors/ext_283.jpg` ở hạng nhất. Sau can thiệp, ảnh đầu tiên là tòa nhà `skyscraper/ext_30.jpg`. [Giả thuyết và nghiên cứu liên quan](CLIP_RETRIEVAL_DIAGNOSIS.md) · [Phép đo trên mẫu và đường dẫn ảnh sai](https://github.com/HieuNM1804/training-free-ZS-SBIR/blob/97a8a4b/docs/results/clip-retrieval-diagnosis.json)

| Trên tất cả sketch | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin |
|---|---:|---:|---:|
| Tỷ lệ lượt sai top 10 tập trung ở 1% ảnh bị chọn sai nhiều nhất, cosine gốc | 99,68% | 97,94% | 97,94% |
| Cùng phép đo sau can thiệp | 41,67% | 31,36% | 45,68% |
| Số sketch có top 1 từ sai thành đúng | 6.349 | 4.944 | 553 |
| Số sketch có top 1 từ đúng thành sai | 128 | 389 | 61 |

Mẫu lỗi này phản ánh sự tập trung bất thường vào một số ảnh gallery. Chỉ trừ độ hút của ảnh, theo đối chứng NNN dùng gallery, đưa precision trên mẫu Sketchy xuống 35,71% và 26,58%. Loại một hướng miền ước lượng từ chênh lệch giữa prompt sketch và photo đạt 45,29% và 37,39%. Loại một hướng ngẫu nhiên đạt 41,08% và 34,34%. Khi giữ danh tính từ qua hai miền, điểm tăng mạnh hơn. Các thí nghiệm ủng hộ sai lệch miền trong phép so trực tiếp và cho thấy dữ liệu vẫn chứa thông tin phân biệt lớp có thể khai thác qua text. [Chẩn đoán NNN và đối chứng hoán vị](https://github.com/HieuNM1804/training-free-ZS-SBIR/blob/97a8a4b/docs/results/clip-retrieval-diagnosis.json) · [Phép chiếu và ví dụ truy hồi](results/reverse-prompt-diffusion/evidence.json)

## Phương pháp

1. **Chỉ mục ảnh gallery.** Dùng 20.931 khái niệm công khai của Open Images V7. Mỗi từ có hai embedding miền, mỗi embedding là trung bình chuẩn hóa của hai câu cố định. CLIP so mỗi ảnh với các prompt photo, lấy tám từ gần nhất và cho trọng số softmax ở nhiệt độ 0,03. Giữ từ và trọng số rồi thay embedding photo bằng embedding sketch tương ứng. Tổng chuẩn hóa là biểu diễn mới của ảnh. Sketch truy vấn giữ nguyên embedding toàn ảnh. [FreeDom, WACV 2025](https://arxiv.org/abs/2412.03297) · [Open Images V7](https://storage.googleapis.com/openimages/web/download_v7.html)
2. **Lan truyền trên các ảnh liên quan.** Từ embedding RGB đã trừ tâm gallery, nối các cặp ảnh cùng nằm trong danh sách 96 láng giềng gần nhất của nhau. Trọng số cạnh là lập phương của cosine dương và được chuẩn hóa theo bậc. Lan truyền riêng cả biểu diễn từ và biểu diễn RGB đã trừ tâm, với hệ số 0,92 trong 12 vòng. Tính các biểu diễn này một lần khi lập chỉ mục. [Iscen và cộng sự, CVPR 2017](https://arxiv.org/abs/1611.05113)
3. **Phản hồi từ các ảnh đầu bảng.** Sketch chấm từng ảnh bằng tích vô hướng với biểu diễn từ đã lan truyền. Chuẩn hóa điểm theo các ảnh của riêng truy vấn, lấy 96 ảnh dẫn đầu, rồi lấy trung bình các embedding RGB đã lan truyền với trọng số softmax ở nhiệt độ 0,6. Điểm cuối gồm 40% điểm lượt đầu và 60% điểm chuẩn hóa từ vector phản hồi. Mỗi sketch được xử lý độc lập. [Chum và cộng sự, ICCV 2007](https://www.robots.ox.ac.uk/~vgg/publications/2007/Chum07b/) · [Mã tính điểm](../tf_sbir/reverse_prompt.py)

Toàn bộ thí nghiệm dùng ảnh RGB toàn khung, hai encoder CLIP đóng băng và suy luận không gradient. Danh mục từ công khai được giữ nguyên cho cả ba split. Các từ, trọng số, láng giềng và ảnh phản hồi đều do CLIP xác định. Tên miền trong câu prompt, nhiệt độ và số láng giềng là lựa chọn thiết kế được công khai. Nhãn lớp dùng để xác định split, chẩn đoán và chấm điểm.

## Kết quả trên toàn bộ gallery

Mỗi ô ghi **precision / mAP**, đơn vị %. Ext-1 và TU dùng P@100 / mAP@all. Ext-2 dùng P@200 / mAP@200.

| Phương pháp | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin |
|---|---:|---:|---:|
| CLIP ViT-B/32 gốc, cosine RGB | 41,64 / 30,29 | 33,85 / 36,34 | 46,26 / 29,07 |
| Cosine RGB và cùng quy trình diffusion, phản hồi, không cần từ điển để chấm điểm | 35,19 / 39,88 | 26,22 / 28,18 | 48,22 / 47,26 |
| Bỏ phiếu từ kiểu FreeDom, bản chuyển sang SBIR trước đây | 43,32 / 43,12 | 39,73 / 40,66 | — |
| Phân bố khái niệm chung và diffusion | 75,96 / 76,37 | 67,89 / 69,16 | 76,77 / 75,62 |
| Nhánh prompt chuyển miền trước đây, `domain-retain0-joint` | 78,15 / 76,18 | 67,03 / 69,25 | 78,81 / 77,97 |
| **Ảnh gallery → prompt sketch, diffusion và phản hồi** | **80,14 / 80,85** | **71,59 / 73,16** | **80,61 / 76,96** |

Bản FreeDom trong repository lấy phiếu từ một sketch và 19 ảnh gallery gần nhất, dùng bảy từ cho mỗi thành viên và giữ bảy từ sau bỏ phiếu. Đây là số chạy lại theo thiết lập SBIR của dự án. [Điểm FreeDom Ext-1](results/reverse-prompt-diffusion/historical/sketchy_1-frequency-transfer.json) · [Điểm FreeDom Ext-2](results/reverse-prompt-diffusion/historical/sketchy_2-frequency-transfer.json)

Cấu hình mới tăng precision lần lượt **1,99 / 4,56 / 1,80 điểm** so với nhánh prompt trước đây. mAP tăng trên hai split Sketchy và giảm **1,00 điểm trên TU** khi tính từ số liệu chưa làm tròn. So với cosine gốc, precision tăng ở 23/25, 21/21 và 30/30 lớp. Trên Ext-1 mức giảm tập trung ở `chicken` (−5,50 điểm) và `scissors` (−2,11). [Kết quả theo lớp](results/reverse-prompt-diffusion/by-class.csv) · [Kết quả nhánh phân bố khái niệm](https://github.com/HieuNM1804/training-free-ZS-SBIR/blob/b4554fd/docs/CALIBRATED_CONCEPT_RESULTS.md)

SeCo-SBIR dùng ViT-B/32 và huấn luyện prompt, adapter cùng tham số liên kết. Bảng 1 của tác giả công bố P là 87,0 / 77,5 / 84,7 và mAP là 80,6 / 80,0 / 78,7. Khoảng cách precision với các mốc đó là 6,86 / 5,91 / 4,09 điểm. Cách chọn cấu hình, pipeline nạp dữ liệu và điều kiện huấn luyện khác nhau giữa hai nghiên cứu. [SeCo-SBIR, Bảng 1](https://arxiv.org/html/2608.03120v1#S4.T1)

## Đối chứng và giới hạn của kết luận

| Thay một thành phần của cấu hình khóa, báo precision (%) | Ext-1 | Ext-2 | TU |
|---|---:|---:|---:|
| Đầy đủ | 80,14 | 71,59 | 80,61 |
| Chuyển sketch → prompt photo, giữ cùng diffusion và phản hồi | 72,13 | 58,97 | 69,66 |
| Giữ chiều gallery → sketch, bỏ cả diffusion và phản hồi | 76,61 | 68,34 | 77,17 |
| Chuyển sketch → photo, bỏ cả diffusion và phản hồi | 69,82 | 57,53 | 67,42 |
| Chỉ bỏ diffusion | 79,80 | 71,38 | 79,71 |
| Chỉ bỏ phản hồi | 79,42 | 70,56 | 79,98 |
| Ảnh chọn từ như cũ, thay prompt sketch bằng prompt photo | 78,82 | 69,19 | 80,35 |
| Giữ cấu trúc nhưng xáo danh tính từ giữa hai bộ prompt | 4,04 | 2,01 | 2,24 |

**Chiều chuyển và danh tính từ có bằng chứng mạnh nhất.** Đổi từ chiều sketch → photo sang chiều gallery → sketch tăng 8,01 / 12,62 / 10,95 điểm precision. Khoảng bootstrap 95% theo lớp lần lượt là 5,28–10,80 điểm, 6,30–18,79 điểm và 6,51–16,39 điểm. Chiều gallery vẫn thắng khi bỏ cả hai bước ngữ cảnh. Phần precision tăng thêm riêng từ diffusion nhỏ và các khoảng theo lớp chứa 0. Lợi ích mAP của diffusion rõ hơn ở Ext-1 và TU, lần lượt 3,52 và 2,51 điểm. Với TU, dùng prompt photo cho cả hai phía vẫn đạt **80,35 P / 77,15 mAP**. Phần lớn lợi ích ở đây vẫn có khi bỏ từ chỉ miền sketch. [99 dòng chấm đầy đủ](results/reverse-prompt-diffusion/configurations.csv) · [So sánh ghép cặp và khoảng bootstrap](results/reverse-prompt-diffusion/evidence.json)

**Ảnh dễ gọi tên hơn sketch là một giải thích có giới hạn.** Sau khi xếp hạng, đối chiếu tên lớp thật với tám từ CLIP đã chọn, chỉ gộp khác biệt dấu cách, gạch ngang và gạch dưới. Tính trên các lớp mà danh mục có tên trùng. Ở Ext-1, ảnh đạt 61,28% và sketch đạt 50,01% (24/25 lớp). Ở Ext-2 là 62,21% và 37,35% (18/21 lớp). **Trên TU, thứ tự đảo lại**, ảnh đạt 48,01% và sketch đạt 62,80% (22/30 lớp). Điều này giới hạn cách giải thích bằng khả năng đọc đúng tên lớp và cũng cho thấy từ điển công khai có chồng lấp tự nhiên với nhãn benchmark. [Kiểm tra từ vựng và thứ hạng](results/reverse-prompt-diffusion/verified-rankings.json)

**Mức mới.** FreeDom đã dùng từ vựng rời rạc để chuyển miền và đã minh họa sketch → photo trên MiniDomainNet. Cross the Gap dùng phép đảo tại phía truy vấn trong thí nghiệm truy hồi ảnh–ảnh. Công trình về hubness cũng đã chỉ ra rằng chiều ánh xạ ảnh hưởng thứ hạng trong zero-shot learning. Đóng góp khả dĩ ở đây là đo sự bất đối xứng giữa hai phía, chỉ ra những ảnh sai thường xuyên được đẩy lên đầu và kiểm chứng cách sửa bằng việc lập chỉ mục từng ảnh qua prompt miền truy vấn. Diffusion và query expansion là kỹ thuật đã có tiền lệ. Bằng chứng hiện tại bao phủ hai bộ dữ liệu sketch → photo. Khả năng áp dụng cho miền khác vẫn là giả thuyết. [FreeDom, phụ lục](https://openaccess.thecvf.com/content/WACV2025/supplemental/Efthymiadis_Composed_Image_Retrieval_WACV_2025_supplemental.pdf) · [Cross the Gap, mục 5.1](https://arxiv.org/html/2502.04263v1#S5.SS1) · [Shigeto và cộng sự, ECML PKDD 2015](https://arxiv.org/abs/1507.00825)

## Thiết lập và kiểm chứng

- Colab L4. Sketchy Ext-1 có 14.179 sketch, 17.100 ảnh, 25 lớp. Ext-2 có 12.694 sketch, 12.552 ảnh, 21 lớp. TU có 2.130 sketch, 27.262 ảnh, 30 lớp. [Môi trường, checkpoint và checksum từ vựng](results/reverse-prompt-diffusion/environment.json)
- Dùng hai split unseen Sketchy để thăm dò và chọn cấu hình. Sau khi chẩn đoán ở 24 sketch/lớp, lượt sàng lọc khác dùng 32 sketch/lớp, 39 cấu hình của nhánh phân bố và 57 cấu hình của nhánh chuyển miền. Chấm danh sách rút gọn trên toàn bộ sketch. Tối đa hóa trung bình precision hai split với trọng số ngang nhau rồi xét trung bình mAP nếu hòa. Khóa hai nhánh, tham số và các đối chứng trước lượt TU của nghiên cứu này. Các kết quả TU lịch sử đã tồn tại trong repository. [Bản khóa giữa hai nhánh](results/reverse-prompt-diffusion/inquiry-joint-freeze.json) · [Các đối chứng đã khóa](results/reverse-prompt-diffusion/freeze.json)
- Các khoảng 95% lấy mẫu lại theo lớp 5.000 lần, giữ nguyên cặp kết quả của cùng một sketch. Khoảng trên Sketchy mô tả biến thiên giữa các lớp ở cấu hình đã chọn và chưa hiệu chỉnh cho việc chọn tham số trên unseen.
- Gọi trực tiếp hàm `_retrieval_metrics` trích nguyên văn từ `D:/distill/src/model.py` cho cosine gốc trên ba split. Với phương pháp mới, chuyển mỗi thứ hạng thành cosine phân biệt và chạy lại toàn bộ **29.003 truy vấn** qua hàm gốc. Giữ nguyên thứ hạng thì độ lệch lớn nhất giữa hai evaluator bằng 0 cho cả AP lẫn precision của từng truy vấn. Khi tính lại bản thân điểm CLIP và diffusion, mức lệch lớn nhất ở **trung bình của một split** dưới 0,002 điểm phần trăm. Các trường top 1 và top 10 trong bảng chẩn đoán dùng thứ tự gallery cố định khi hòa điểm. [Bản kiểm chứng đầy đủ](results/reverse-prompt-diffusion/verified-rankings.json)
- Phiên bản chấm điểm là `d343910` cho nhánh chính và `1b8d80a` cho nhánh phân bố. Trên runtime mới có cùng dữ liệu và checkpoint, chạy bốn stage `pilot`, `sketchy`, `controls`, `freeze` trong từng worktree bằng `tf_sbir.reverse_benchmark` hoặc `tf_sbir.concept_benchmark`. Sau khi cả hai nhánh hoàn thành, chạy `tf_sbir.inquiry_joint_freeze` ở nhánh báo cáo, rồi `heldout` ở hai worktree đã khóa. [Mã benchmark](../tf_sbir/reverse_benchmark.py) · [Mã khóa giữa hai nhánh](../tf_sbir/inquiry_joint_freeze.py) · [Mã kiểm chứng evaluator](../tf_sbir/verify_locked_ranks.py)
