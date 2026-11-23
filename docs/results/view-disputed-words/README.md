# So sánh hai hướng nhìn ở đầu vào bộ truy hồi

Các điểm dưới đây là phần trăm. Ext-1 dùng mAP toàn gallery và P@100. Ext-2 dùng mAP@200 và P@200. TU-Berlin dùng mAP toàn gallery và P@100. GPU Colab là NVIDIA L4. Backbone là OpenAI CLIP ViT-B/32 đóng băng. Mỗi sketch được xử lý độc lập. Nhánh thí nghiệm dùng từ điển Open Images công khai gồm 20.931 từ. Việc chọn hướng trên hai split Sketchy unseen mang tính thăm dò.

## Đối chứng mạnh đã khóa

| Nguồn bằng chứng vào bước lan truyền RGB | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin 30 lớp |
|---|---:|---:|---:|
| Sketch gốc | 85,12 / 84,47 | 77,31 / 75,26 | 80,28 / 80,91 |
| Trung bình xác suất của sketch gốc và bản lật ngang | 85,29 / 84,65 | 77,55 / 75,45 | 80,59 / 81,32 |

Thành phần thay đổi là phân bố từ trước khi lan truyền trên gallery. Cấu hình dùng cây nhóm ảnh đã khóa ở nhánh nghiên cứu trước. Nó vẫn đọc vùng sketch trong thành phần điểm thứ hai. Đối chứng cũ và mới được chấm lại với hàm KD-SBIR gốc trên mọi truy vấn. Rank replay bảo toàn thứ tự ảnh. Với Ext-2, cả hai metric chỉ sử dụng 200 hạng đầu. Sai số tuyệt đối lớn nhất giữa phép tính theo hạng và replay dưới 0,0000003. Số lớp tăng AP trung bình là 19/25, 16/21 và 20/30. Hai split Sketchy có thể trùng lớp nên cần thận trọng khi diễn giải sự nhất quán giữa chúng. Khoảng bootstrap trong JSON được tính trên các lớp, không phải trên từng ảnh.

## Giả thuyết kiểm tra

Gọi a và b là xác suất của một từ ở hai hướng toàn ảnh, c là xác suất trung bình của từ đó trên 13 crop ở hai hướng. Thí nghiệm chính lấy trung vị của a, b và c, sau đó chuẩn hóa theo từ. Cách này cho phép bằng chứng vùng phân xử khi hai hướng bất đồng. Đối chứng trộn crop trực tiếp dùng (a+b+c)/3. Đối chứng tráo chuyển các giá trị c giữa các từ lân cận trong bảng xếp hạng xác suất toàn ảnh, theo từng khối 16 từ.

| Nhánh từ điển trực tiếp | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin |
|---|---:|---:|---:|
| Trung bình hai hướng | 76,22 / 80,02 | 72,45 / 71,33 | 74,13 / 78,28 |
| Trung vị với quan sát crop | 75,80 / 79,76 | 72,08 / 70,94 | 73,58 / 77,52 |
| Trung bình thêm quan sát crop | 76,59 / 80,88 | 73,07 / 71,85 | 74,46 / 78,74 |

Trung vị thấp hơn trung bình hai hướng trên cả ba lượt đánh giá. Khi chỉ giữ phần biến động của c nằm ngoài khoảng từ min(a,b) đến max(a,b), điểm mAP trực tiếp trên hai split Sketchy là 76,61 và 73,04. Việc thêm bằng chứng này vào bộ xếp hạng mạnh cần đối chứng riêng. Trên hai split, phương pháp dùng trung vị hay thêm crop nguyên vẹn ở đầu vào diffusion đều thua cách chỉ dùng trung bình hai hướng. Tráo bằng chứng crop còn cho điểm cao hơn crop đúng trên bộ xếp hạng mạnh. Chưa có bằng chứng để lấy cơ chế phân xử theo từ làm đóng góp chính.

## Tái lập

Mã chấm trực tiếp khóa ở commit 5b5b091. Mã thay tín hiệu đầu vào của bộ xếp hạng mạnh khóa ở 1c3e085. Tùy chọn chấm đầy đủ cặp đối chứng khóa ở 1da45bd. Chạy module trên Colab có dữ liệu, cache và evaluator gốc của dự án.

```bash
python -m tf_sbir.view_disputed_words --dataset sketchy --official
python -m tf_sbir.view_disputed_words --dataset tuberlin --official
python -m tf_sbir.view_disputed_words --dataset sketchy --strong --primary --official
python -m tf_sbir.view_disputed_words --dataset tuberlin --strong --primary --official
```

Các file có official trong tên chứa kết quả KD-SBIR gốc cho mọi truy vấn. File sketchy-decomposed, sketchy-internal và sketchy-strong lưu các phép kiểm tra cơ chế; thứ hạng được đối chiếu với KD-SBIR trên một truy vấn mỗi batch cho từng cấu hình. Phép trung bình các dự đoán dưới nhiều biến đổi đã có tiền lệ ở nghiên cứu thích ứng lúc suy luận. Thí nghiệm ở đây dùng suy luận đóng băng. [MEMO, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/fc28053a08f59fccb48b11f2e31e81c7-Abstract-Conference.html)
