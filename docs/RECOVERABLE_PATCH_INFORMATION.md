# Chi tiết bị phép chiếu patch của CLIP bỏ qua có gây lỗi truy hồi không?

Mỗi patch của OpenAI CLIP ViT-B/32 gồm 3.072 giá trị, còn phép chiếu đầu vào chỉ trả về 768 số. Một không gian bị bỏ qua đã được mô tả trong nghiên cứu về độ bền của ViT. Câu hỏi ở đây hẹp hơn. Trên chính những ảnh bị truy hồi sai, phần bị bỏ qua có mang bằng chứng phân biệt ảnh đúng với ảnh sai không? Một tín hiệu đo được ở tầng đầu vẫn có thể vô ích đối với xếp hạng.

## Can thiệp

1. Gọi W là bộ trọng số convolution 32×32, u là hướng thay đổi ba kênh RGB cùng một lượng trong tọa độ đã chuẩn hóa của CLIP. Lập ma trận gồm W chiếu trên u và một hàng hằng để bảo toàn trung bình mỗi patch. SVD cho cơ sở trực chuẩn Q của hạt nhân. Chiếu ảnh thật lên Q thu được n. Phép đối xứng x′ = x − 2n giữ màu tương đối, trung bình và độ lớn của phần chi tiết trong từng patch. Trong số học chính xác, W(x′) = W(x). Một số pixel của x′ có thể nằm ngoài khoảng hiển thị. Báo tỷ lệ này cùng sai số FP16 đọc trực tiếp từ encoder.
2. Dời lưới theo bốn độ lệch (±8,±8) pixel, dùng chính W để đọc n trên mỗi lưới. Đưa token về lưới gốc theo tỷ lệ diện tích giao nhau của hai patch rồi lấy trung bình. Gọi phần quan sát mới là D, đã trừ Wn. Biên tuần hoàn là một lựa chọn thực nghiệm. Kiểm tra phép đưa token về lưới gốc bằng tích phân trên bản đồ pixel đã bung từ token.
3. Đặt W(x) + αD ở đầu transformer, giữ mọi trọng số và vị trí gốc của CLIP. Tính thay đổi cosine RGB hoặc thay đổi log-bằng-chứng qua 64 từ của đối chứng vùng đã khóa. Cộng thay đổi vào cùng ma trận điểm sketch–ảnh. Mỗi sketch được xử lý độc lập. Công thức can thiệp cũng áp dụng nguyên dạng cho ảnh màu.

## Đối chứng và trình tự

Thử α thuộc {−8, −2, 0,5, 2, 8, 24}; α = 0 cho đối chứng vùng. Mỗi phương án đọc dùng thay đổi RGB, từ công khai hoặc trung bình hai tín hiệu, với hệ số 0,15, 0,6 hoặc 1,5. Chạy cùng lưới cho năm đối chứng. Một đối chứng tráo và đổi dấu các hệ số ngay trong hạt nhân, một đối chứng đặt bằng chứng ở ô lân cận. Hai đối chứng đọc phần đã nhìn thấy hoặc toàn bộ ảnh trên cùng các lưới lệch, khớp độ lớn token. Đối chứng cuối dùng độ lớn tự nhiên của toàn bộ ảnh. Khi tráo hệ số, giữ nguyên chuẩn của từng patch và khớp thêm chuẩn thay đổi token trên từng ảnh.

So sánh phương án thật với từng đối chứng tại cùng α, cách đọc và hệ số. Đồng thời cho từng đối chứng chọn cấu hình riêng theo trung bình mAP của hai split Sketchy với trọng số bằng nhau. Chạy một pilot 36 sketch mỗi lớp, giữ toàn bộ gallery, phát lại thứ hạng từng ứng viên cuối qua hàm gốc của KD-SBIR. Ghi bao nhiêu top-1 được sửa và bị hỏng cùng khoảng bootstrap ghép cặp theo lớp. Đây là lựa chọn thăm dò trên unseen. Nếu lưới lệch thông thường hay hệ số tráo cho lợi ích tương tự, thứ hạng chưa quy được cho nội dung bị bỏ qua.

Một phép đo bổ sung dùng các cặp ảnh đúng và sai lấy từ thứ hạng của đối chứng đã khóa. Đọc độc lập hai phía để đo thay đổi do riêng sketch, riêng ảnh, hoặc cả hai, trên cùng cặp ứng viên. Phân biệt cải thiện thứ hạng trên cặp khó với mAP trên toàn gallery. Nhãn chỉ dùng cho đánh giá sau xếp hạng. Chỉ chấm chuyển giao TU-Berlin khi có lý do giữ cơ chế và đã khóa cấu hình.

## Công trình liên quan và giới hạn novelty

Liu và cộng sự đã phân tích hạt nhân của token embedding và tận dụng không gian gần hạt nhân để fine-tune ViT cho robustness. Rojas-Gomez và cộng sự đã dùng các pha của lưới để làm ViT tương thích hơn với phép tịnh tiến. Đóng góp đang kiểm tra là truy riêng phần tín hiệu tự nhiên vô hình ở lưới gốc, đọc lại nó bằng các trọng số đã có và đo tác động lên từng quyết định truy hồi. Hiệu quả của can thiệp là câu hỏi thực nghiệm. [Liu và cộng sự, CPAL 2025](https://openreview.net/pdf/83f6201771fc4c4a325cd869d18feea0b36de419.pdf) · [Rojas-Gomez và cộng sự, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Rojas-Gomez_Making_Vision_Transformers_Truly_Shift-Equivariant_CVPR_2024_paper.pdf)

Đối chứng nền dùng ánh xạ qua từ công khai và diffusion trên gallery RGB đã khóa. Trong tối đa ba nguồn cơ chế trực tiếp của cấu hình này, hai nguồn thuộc nền đã kế thừa và nguồn thứ ba là phân tích hạt nhân đầu vào. Các lưới, phép đối xứng, cách can thiệp và đối chứng được mô tả riêng để có thể kiểm tra đóng góp tăng thêm.
