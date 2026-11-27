# Căn theo các patch giúp khớp phản ứng của block 6, nhưng lợi thế theo lớp gần bằng 0

Hai split Sketchy có lần lượt 800 và 672 ảnh nguồn, mỗi ảnh có hai bản vẽ thật của hai lượt vẽ. Mỗi lớp có 32 ảnh nguồn. Tổng cộng có 9.600 và 8.064 cặp sketch–ảnh, mỗi cặp đo ở hai nửa lưới hoán đổi. Tập này dùng để chẩn đoán cơ chế. Nhãn và ảnh nguồn chỉ dùng để lập phép đối chiếu. [Giao thức đã khóa trước lần đo](../../native-block-depiction.md)

Mỗi ô là cosine giữa phần thay đổi do block tạo ra trên các patch giữ lại, sau khi trừ trung bình theo vị trí của từng hình. Cột “giữ nguyên” đọc các kênh trong cùng không gian con phía ảnh. Cột “tráo” giữ nguyên tập patch dùng để ước lượng phép căn và tráo thứ tự ghép chúng. Các giá trị là trung bình cân bằng theo lớp, hai bản vẽ và hai nửa lưới.

| Cặp so sánh | Ext-1 giữ nguyên | Ext-1 căn đúng | Ext-1 tráo | Ext-2 giữ nguyên | Ext-2 căn đúng | Ext-2 tráo |
|---|---:|---:|---:|---:|---:|---:|
| Bản vẽ và đúng ảnh nguồn | 0,13952 | 0,21361 | −0,02229 | 0,14649 | 0,22647 | −0,02166 |
| Bản vẽ và ảnh khác vật thể, cùng lớp | 0,09378 | 0,18213 | −0,01951 | 0,10122 | 0,19415 | −0,01975 |
| Bản vẽ và ảnh khác lớp | 0,08444 | 0,17267 | −0,01952 | 0,08924 | 0,18085 | −0,01901 |

Lợi thế của ảnh cùng lớp so với khác lớp tăng **0,00012** trên Ext-1, khoảng bootstrap 95% theo lớp **[−0,00486, 0,00490]**. Ext-2 tăng **0,00131 [−0,00873, 0,01264]**. Khi chỉ tính các truy vấn thuộc lớp riêng của mỗi split, mức tăng là 0,00188 [−0,00346, 0,00690] và 0,00377 [−0,00813, 0,01732]. Giữ nguyên trung bình không gian của phản ứng cũng cho hai khoảng chứa 0. Trên ảnh nguồn thực, căn đúng vượt đọc nguyên ở 25/25 và 21/21 lớp.

Phép căn truy cập cả hai ảnh khi ước lượng và có nhiều thông tin hơn phương pháp dự kiến dùng lúc truy hồi. Ngay cả trong điều kiện đó, mức tiến bộ ở vật thể khác cùng lớp gần trùng với mức tăng ở ảnh sai lớp. Kết quả chưa ủng hộ việc viết bộ truy hồi dùng phép căn này. Chưa có mAP hay kết quả TU-Berlin cho ứng viên.

Kaggle, bốn NVIDIA L4, OpenAI CLIP ViT-B/32, seed 42. Ghi đầu vào và đầu ra trùng với hook của forward gốc trong phép kiểm tra ở cả bốn shard. Đối chứng biến đổi trực giao đã biết cho sai số tối đa 2,23 × 10⁻¹⁶. Kích hoạt gốc dùng FP32, phép căn và cosine chẩn đoán dùng FP64. Giao thức và mã trích xuất ở commit `a54d0e2`, đo các cặp ở `2aa3048`, tổng hợp ở `4b4f431`. [Số liệu theo lớp và khoảng tin cậy](summary.json) · [Điểm từng cặp, fold và manifest kiểm tra mã băm](paired-scores-L4.zip)

## Phần thông tin bị phép chiếu cuối của CLIP loại bỏ

Phép chiếu thị giác từ 768 xuống 512 chiều có hạng 512 trên đúng checkpoint này. Với từng sketch, giữ nguyên hai photo đã được Semantic Dot chọn trên toàn gallery sau khi loại ảnh nguồn. Một photo dẫn đầu trong lớp đúng, photo kia dẫn đầu trong các lớp sai. Bảng sau ghi tỷ lệ phần trăm chọn đúng giữa chính hai photo đó. Tất cả sketch unseen đều có mặt. Mỗi hàng được đo độc lập. [Mô tả phép đo và đối chiếu với IsoCLIP](../../native-projection-evidence.md)

| Đọc CLIP thị giác | Ext-1 toàn bộ | Ext-2 toàn bộ | Ext-1, 4.232 lỗi Semantic | Ext-2, 4.463 lỗi Semantic |
|---|---:|---:|---:|---:|
| Đầu ra CLIP gốc | 56,58 | 55,18 | 40,69 | 45,87 |
| Toàn bộ 768 chiều trước phép chiếu | 57,78 | 53,80 | 46,79 | 45,57 |
| 512 chiều được giữ, bỏ hệ số co giãn của phép chiếu | 57,78 | 54,16 | 46,72 | 44,01 |
| 256 chiều bị loại | 52,54 | 51,97 | 46,20 | 49,54 |
| 256 chiều ngẫu nhiên từ phần được giữ | 57,77 | 55,21 | 45,63 | 41,21 |

Ở Ext-2, trên các lỗi của Semantic, phần bị loại hơn đối chứng ngẫu nhiên 8,34 điểm phần trăm khi mỗi truy vấn có trọng số bằng nhau. Khi mỗi lớp có trọng số bằng nhau, mức chênh là +6,48 điểm với khoảng bootstrap 95% [−1,24, +14,85]. Ở Ext-1, chênh lệch theo lớp là +0,66 [−1,97, +3,05]. Trên toàn bộ cặp, phần bị loại đúng ở 52,54% và 51,97%, thấp hơn đối chứng cùng số chiều ở cả hai split. Có những truy vấn mà phần bị loại chọn đúng, song hiệu ứng so với đối chứng thiếu nhất quán giữa các lớp và split. Dữ liệu hiện tại chưa ủng hộ việc xây phương pháp chỉ từ 256 chiều này.

Sai số tuyệt đối tối đa khi phát lại biên cosine CLIP gốc là 0,000925. Chuẩn tương đối của phép chiếu 256 trục bị loại qua đầu ra gốc là 9,28 × 10⁻¹⁶ trở xuống. Báo cáo này dùng đúng các cặp từ lần đánh giá baseline trước. Đo trên cặp đã chọn không cung cấp mAP hay P@K. [Số từng nhóm và khoảng tin cậy](projection-evidence.json) · [Các cặp, biên điểm và mã băm từ Kaggle](projection-pairs-L4.zip)

## Bản vẽ khác của cùng vật thể có lặp lại lỗi không?

Giữ lại những truy vấn Semantic Dot xếp sai hai ảnh đã cố định, ảnh nguồn ưu tiên ứng viên đúng và có ít nhất hai bản vẽ khác cùng nguồn. Có 3.745 và 3.960 truy vấn thỏa điều kiện, từ 1.539 và 1.514 ảnh nguồn. Với mỗi truy vấn, lần lượt lấy một hoặc hai bản vẽ khác cùng nguồn. Đối chứng dùng một ảnh nguồn khác trong cùng lớp rồi lấy cùng số bản vẽ của ảnh đó. Hai bên chấm đúng cùng cặp ứng viên. Mỗi nguồn có cùng trọng số trong lớp, mỗi lớp có cùng trọng số trong split.

| Xác suất lặp lại lựa chọn sai | Ext-1 cùng nguồn | Ext-1 nguồn khác cùng lớp | Ext-2 cùng nguồn | Ext-2 nguồn khác cùng lớp |
|---|---:|---:|---:|---:|
| Một bản vẽ khác chọn cùng ảnh sai | 21,13% | 15,82% | 23,66% | 19,47% |
| Hai bản vẽ khác đều chọn cùng ảnh sai | 10,52% | 7,66% | 12,32% | 10,25% |

Với hai bản vẽ, chênh lệch cùng nguồn trừ đối chứng là +2,86 điểm phần trăm [1,95, 3,86] và +2,07 [1,07, 3,06]. Có thành phần lặp lại riêng theo ảnh nguồn, song trong nhóm đã chọn, một bản vẽ khác thường dẫn tới lựa chọn khác với bản vẽ lỗi. Số đo chưa xác định cử chỉ, chi tiết hay tác nhân nào gây đổi hạng. Các bản vẽ khác và nhãn được sử dụng hồi cứu để so sánh.

Ví dụ trên Ext-2, ba bản vẽ cùng con tê giác cho biên điểm ảnh tê giác trừ ảnh bò lần lượt −0,064, +0,414 và +0,216. Hai ảnh ứng viên giống hệt nhau ở cả ba lần đo. Ảnh nguồn cho biên +0,904. Ví dụ thuộc cặp lớp thường gặp nhất trong 1.193 trường hợp đủ điều kiện có toàn bộ bản vẽ khác nghiêng về đúng lớp. Chọn trường hợp ở trung vị biên điểm của 53 truy vấn tê giác–bò trong nhóm đó. [Ảnh nguồn, đường dẫn các bản vẽ và biên điểm](redraw-example.json) · [Kết quả tổng hợp](redraw-error-specificity.json) · [Giá trị trên từng truy vấn](redraw-errors-L4.zip)

Một câu hỏi còn mở là liệu một lần quan sát có đủ để phân biệt cách thể hiện và nội dung khi người thể hiện có thể chọn những chi tiết khác nhau. Điều kiện kiểm chứng có ý nghĩa là dự báo được cách các bản vẽ giữ lại thay đổi quyết định giữa hai ảnh thật. Một mô hình chỉ ước lượng độ bất định chung hoặc thay đổi điểm xếp hạng có nhiều tiền lệ. PCME đã học embedding xác suất cho retrieval, Probabilistic U-Net đã đánh giá phân phối các chú giải khác nhau của cùng một ảnh, và trong repo một phép dự báo độ ổn định của thứ hạng đã cải thiện AUROC trong khi mAP gần như giữ nguyên. Chứng cứ về một cơ chế training-free phục hồi *quyết định đúng* từ một hình vẫn còn thiếu. [PCME, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Chun_Probabilistic_Embeddings_for_Cross-Modal_Retrieval_CVPR_2021_paper.html) · [Probabilistic U-Net, NeurIPS 2018](https://proceedings.neurips.cc/paper_files/paper/2018/hash/473447ac58e1cd7e96172575f48dca3b-Abstract.html) · [Đối chứng AUROC và mAP trong repo](https://github.com/HieuNM1804/training-free-ZS-SBIR/blob/research/redraw-rank-stability/docs/results/redraw-rank-stability/README.md)
