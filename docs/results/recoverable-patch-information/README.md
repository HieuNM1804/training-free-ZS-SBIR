# Chi tiết bị bỏ qua ở phép chiếu patch và quyết định truy hồi

Trên đúng checkpoint OpenAI CLIP ViT-B/32, phép chiếu từ 1.024 bậc tự do độ sáng của mỗi patch, có thêm ràng buộc giữ trung bình, có hạng 769. Hạt nhân còn 255 chiều. Năng lượng độ sáng sau trừ trung bình nằm trong hạt nhân đạt 1,12% trên pilot Sketchy Ext-1 và 1,08% trên Ext-2. Khi đọc lại bằng bốn lưới lệch, chuẩn phần token thu hồi đạt 0,52% và 0,50% chuẩn token gốc. Đối xứng phần vô hình giữ token của phép chiếu FP32 với sai số tương đối tối đa 0,00000272 trong các kiểm tra sketch. Khoảng 15–16% giá trị kênh của hình đối xứng ra ngoài khoảng [0,1], vì vậy đây là đối chứng đại số. [Giao thức và công trình liên quan](../../RECOVERABLE_PATCH_INFORMATION.md)

Mỗi cột là mAP / precision (%), trên 36 sketch mỗi lớp và toàn gallery. Ext-1 dùng 900 sketch, 17.100 ảnh, mAP@all / P@100. Ext-2 dùng 756 sketch, 12.552 ảnh, mAP@200 / P@200. Dò các tham số ngay trên hai split unseen là thiết lập thăm dò. Các cấu hình cuối phát lại toàn bộ thứ hạng qua đúng evaluator của KD-SBIR, sai số từng truy vấn dưới 0,000001. Mọi lượt chạy trên Colab T4.

| Can thiệp trên sketch | Sketchy Ext-1 | Sketchy Ext-2 |
|---|---:|---:|
| Nền vùng, từ điển và gallery RGB | 84,64 / 83,94 | 80,64 / 78,38 |
| Dùng hệ số thật | 85,08 / 84,98 | 80,70 / 78,24 |
| Tráo hệ số và đổi dấu trong hạt nhân, chuẩn khớp | 84,94 / 84,86 | 80,83 / 78,47 |
| Đặt sai ô trên lưới token, cùng tham số | 84,99 / 84,80 | 80,64 / 78,13 |
| Phần toàn ảnh trên lưới lệch, khớp chuẩn và tự chọn cấu hình | 84,79 / 84,10 | 80,76 / 78,49 |

Cấu hình được chọn dùng α = −2, log-bằng-chứng qua từ công khai và hệ số 0,6. Cấu hình tráo tự chọn đúng cùng các tham số. Chênh mAP của hệ số thật so với nền là +0,44 [−0,23; +1,36] trên Ext-1 và +0,05 [−1,16; +1,28] trên Ext-2 theo bootstrap ghép cặp theo lớp. Số top-1 sửa/hỏng là 14/4 và 15/16. Trên trung bình hai split, lợi thế của hệ số thật so với hệ số tráo chỉ khoảng 0,006 điểm mAP. Vì vậy chưa thể quy cải thiện cho ý nghĩa ngữ nghĩa của chi tiết bị bỏ qua.

Thử nghiệm cặp ảnh tách tác động của việc đổi riêng sketch, riêng ảnh và phần tương tác giữa hai đầu. Trong 145 và 161 truy vấn có top-1 sai, tương tác giữa hai thay đổi cosine đẩy ảnh đúng lên so với ảnh đối thủ trên 50,3% và 49,7% truy vấn khi dùng α = −2. Dùng thay đổi cả hai đầu qua từ công khai với cùng hai ứng viên cho tỷ lệ 62,8% và 57,8%. Đối chứng tráo cũng đạt 60,0% và 59,0%. Các tỷ lệ này đo hướng thay đổi khoảng cách điểm trên một cặp đã cố định. Chúng không biểu thị mAP trên toàn gallery.

Một hiệu ứng khác đáng đo riêng xuất hiện khi đưa vào hai biến đổi bằng nhau, trái dấu. Gọi e và o là nửa tổng và nửa hiệu của hai thay đổi embedding. Tỷ trọng trung bình ‖e‖²/(‖e‖² + ‖o‖²) đạt 98,2% trên 1.644 sketch duy nhất và 93,2% trên 2.529 ảnh thuộc tập chẩn đoán. Tỷ trọng này nói rằng embedding chủ yếu thay đổi theo thành phần chung cho hai dấu. Nó không cho biết thứ hạng có đúng hơn hay thành phần đó xuất hiện ở tầng nào. Nhánh chẩn đoán trung gian kiểm tra riêng câu hỏi này. [Thí nghiệm theo tầng](https://github.com/HieuNM1804/training-free-ZS-SBIR/tree/research/patch-response-mediation)

Hai split Sketchy vẫn ở mức pilot. Chưa chấm hướng này trên TU-Berlin. Các bảng JSON và ZIP đi kèm giữ nguyên từng truy vấn, từng phép can thiệp và mã băm từ Colab.
