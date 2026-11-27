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
