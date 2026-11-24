# Những đường truyền nào giúp hai phần sketch cùng gợi ra đúng ảnh?

Giữ đúng 42 patch, cùng vị trí và checkpoint. Hai phần có thể được đọc chung bởi token tổng hợp CLS của CLIP qua cả 12 layer, kể cả khi từng patch chỉ được nhận thông tin từ bên mình. Bật thêm đường phản hồi từ CLS tới patch, rồi bật đường truyền trực tiếp giữa các patch ở hai phần. Với từng cách đọc, chấm mọi ảnh gallery bằng evaluator KD-SBIR, frozen OpenAI CLIP ViT-B/32 và bộ từ Open Images cố định. Có 2.180 / 1.828 sketch và 17.100 / 12.552 ảnh ở hai split.

| Các đường truyền được mở | Ext-1 mAP@all / P@100 | Ext-2 mAP@200 / P@200 |
|---|---:|---:|
| Đọc hai phần riêng rồi cộng đặc trưng đã trừ nền CLS | 47,65 / 50,63 | 45,51 / 44,18 |
| Một CLS đọc cả hai phần ở mỗi layer, patch mỗi bên vẫn độc lập | 49,16 / 51,48 | 47,33 / 45,81 |
| CLS chung còn truyền ngược thông tin về patch mỗi bên | 49,58 / 52,36 | 47,40 / 45,99 |
| Mở thêm tương tác trực tiếp giữa patch hai bên, khôi phục CLIP gốc trên 42 patch | **65,22 / 69,01** | **62,47 / 61,06** |

Khi mở đường patch–patch trực tiếp, mAP tăng **15,64 / 15,07 điểm**. Khoảng bootstrap 95% với trọng số bằng nhau theo lớp là [12,66, 18,74] / [10,21, 20,78]. AP tăng ở 25/25 và 20/21 lớp. Với hai thay đổi trước đó, cả bốn khoảng bootstrap của chênh lệch AP đều chứa 0. Các token riêng của từng phần tái lập lần chạy riêng với sai khác tuyệt đối lớn nhất dưới 0,0007 trên đặc trưng đã chuẩn hóa trong batch kiểm tra. mAP của hàng cuối tái lập phép thử trước trong 0,002 điểm trên mỗi split.

Phát hiện này định vị được chỗ mất bằng chứng trong phép thử chia vùng: chỉ cho một bộ đọc mạnh nhìn thấy đồng thời hai nhóm patch đã xử lý độc lập vẫn làm mất phần lớn lợi ích đọc chung. Các hiệu ứng gắn với thứ tự mở đường truyền và cách attention chuẩn hóa lại khi đổi tập token nhìn thấy. Để xây dựng phương pháp truy hồi, còn phải chứng minh tương tác giữa hai miền có thể cung cấp bằng chứng bổ sung về ảnh đúng. CaiT là tiền lệ về việc tách xử lý patch và đọc vào class token. [CaiT, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Touvron_Going_Deeper_With_Image_Transformers_ICCV_2021_paper.pdf)

Chạy Colab T4, seed 42, mã `002d841`. [Đối chứng và cách khóa các đường truyền](protocol.md) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [AP và precision theo truy vấn](paired-T4.zip)
