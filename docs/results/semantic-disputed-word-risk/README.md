# Kiểm tra nhiễu của 20.931 chiều trong nhánh S

Chấm riêng S trên Colab T4 với seed 42, CLIP ViT-B/32 đóng băng và hàm KD-SBIR gốc cho từng truy vấn. Ext-1 có 14.179 sketch, Ext-2 có 12.694 sketch, TU-Berlin có 2.130 sketch thuộc 30 lớp. Mã chạy ở `09d97f7`. Chọn 0,5 từ ba mức đã đăng ký trước trên hai split Sketchy unseen rồi khóa trước TU-Berlin. [Protocol](protocol.md) · [Khóa](freeze.json)

| mAP / precision (%) | Ext-1 | Ext-2 | TU-Berlin |
|---|---:|---:|---:|
| S gốc | 76,22 / 80,02 | 72,45 / 71,33 | 74,13 / 78,28 |
| Cosine của hai vector softmax | 71,98 / 76,09 | 68,00 / 66,95 | 71,11 / 75,32 |
| S chỉ đọc 64 từ đầu của sketch | 74,51 / 78,39 | 70,72 / 69,53 | 73,52 / 77,41 |
| Trung bình hai lượt đọc phân bố từ của photo | 76,20 / 79,97 | 72,78 / 71,74 | 74,52 / 78,46 |
| Giảm tối đa overlap trong phần khối lượng bất đồng prompt | 76,16 / 80,15 | 72,85 / 71,79 | 74,20 / 78,36 |
| Giảm một nửa phần bất đồng trước khi đọc sketch | 76,27 / 80,05 | 72,61 / 71,55 | 74,32 / 78,41 |
| Giảm cùng lượng khối lượng ở mọi vị trí có thể | 70,45 / 80,95 | 73,74 / 71,22 | 65,93 / 77,44 |
| S với prior khái niệm MLE từ gallery | **81,48 / 82,48** | **74,85 / 73,80** | **77,30 / 77,83** |

## Giải thích lỗi

Trên các truy vấn mà S chọn sai ảnh hạng nhất, lấy một cặp cố định gồm ảnh sai đứng đầu và ảnh đúng điểm cao nhất. Từ đóng góp nhiều nhất chiếm trung vị **83,7 / 77,3 / 78,7%** overlap của ảnh sai và **60,0 / 57,1 / 57,8%** overlap của ảnh đúng. Phần từ ngoài 64 từ đầu của sketch chiếm **3,4 / 2,8 / 3,1%** overlap ở ảnh sai và **22,7 / 14,9 / 14,0%** ở ảnh đúng. Chỉ dùng top 64 sửa thứ tự của 98/3.520, 97/4.173 và 11/530 cặp cố định. Đây là phân tích sau khi chấm với nhãn đánh giá, trên các cặp cụ thể trong lỗi hạng nhất.

Phần overlap bất đồng khi đọc cùng photo với prompt miền sketch thường lớn ở ảnh đúng hơn ảnh sai trên hai split Sketchy. Trong số các cặp lỗi được xét, ảnh sai có tỷ lệ bất đồng cao hơn ảnh đúng ở 41,8 / 43,7 / 52,1% trường hợp. Phương pháp dùng ngân sách phụ thuộc prompt thay đổi mAP −0,05 / +0,40 / +0,07 điểm. So với hai đối chứng bỏ cùng lượng xác suất, kết quả đổi chiều giữa các split. Phép cắt đuôi và L2 đều làm giảm cả hai metric trên cả ba split.

Prior MLE là đối chứng có từ trước. Nó cải thiện mAP +5,27 / +2,40 / +3,17 điểm, với khoảng bootstrap theo lớp 95% lần lượt là [3,64, 6,99], [0,77, 4,02] và [1,59, 4,78]. Precision TU-Berlin thay đổi −0,45 điểm.

Điểm robust là cận dưới từ 128 từ đầu và cận cho phần còn lại; khoảng sai số được kiểm tra trên bài toán đầy đủ cho bốn truy vấn và 64 ảnh mỗi split. Sai lệch trung bình tuyệt đối khi chạy lại S so với lượt đã lưu dưới 0,005 điểm ở cả hai metric.

[Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU-Berlin](tuberlin.json) · [Số liệu từng truy vấn](paired-T4.zip)
