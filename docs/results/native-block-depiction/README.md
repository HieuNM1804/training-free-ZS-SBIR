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
