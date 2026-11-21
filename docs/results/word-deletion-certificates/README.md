# Một từ có thể đổi ảnh đầu danh sách như thế nào?

Vô hiệu hóa riêng bằng chứng của sketch cho một từ rồi tính lại **toàn bộ** phản hồi của gallery. Cả ảnh và bộ từ vựng gallery được giữ cố định. Một ảnh có thể đổi thứ hạng qua hai đường: từ đó đóng góp trực tiếp vào điểm của ảnh, và từ đó từng góp phần khiến các ảnh khác trở thành bằng chứng cho ứng viên.

Với điểm xác nhận theo từ $S_i(l)=l^\top A_i l$, đặt $H=A_i-A_j$ cho một cặp ảnh. Khi bỏ từ $k$, mức giảm chính xác của chênh lệch điểm là $l_k[(H+H^\top)l]_k-l_k^2H_{kk}$. Tính đồng thời các mức giảm qua $B$ và $B^\top$ giúp tìm từ có thể đảo hạng. Các hệ số của $A_i$ đều không âm theo công thức lấy mẫu hai ảnh. Vì thế nếu mức giảm của ảnh đứng nhất nhỏ hơn khoảng cách ban đầu với ảnh thứ hai, từ đó không thể đẩy bất kỳ ảnh nào lên hạng nhất. Với mỗi từ còn lại, chương trình chấm lại toàn bộ gallery. [Công thức, chứng minh và thiết kế](../../WORD_DELETION_CERTIFICATE.md)

## Kiểm tra trên mọi truy vấn

| Số sketch có thể đảo thứ tự cặp ảnh đầu | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin |
|---|---:|---:|---:|
| Công thức hai đường theo đúng biến thiên điểm | 6.088 | 6.459 | 1.075 |
| Bỏ từ đứng đầu theo likelihood của sketch | 1.684 | 1.485 | 314 |
| Chỉ xét đường đóng góp trực tiếp | 6.064 | 6.339 | 1.074 |
| Số sketch đã kiểm | 14.179 | 12.694 | 2.130 |

Một phép đảo trong bảng yêu cầu chênh lệch sau can thiệp vượt ngưỡng số học theo hướng ngược lại. Chỉ số kiểm tra cặp ảnh đầu khác với thay đổi danh tính ảnh đứng nhất khi cho các ảnh ngoài cặp cùng cạnh tranh. Quy tắc dựa trên đóng góp trực tiếp có thể tìm được nhiều lần đảo. Công thức hai đường cho biết khi nào một cặp chịu được **mọi** phép bỏ một từ và giải thích phần phản hồi được tính lại.

## Kiểm tra toàn gallery

Chọn bốn sketch mỗi lớp bằng seed cố định. Đánh giá toàn bộ khả năng bỏ một trong 20.931 từ đối với ảnh hạng nhất, sau đó tra nhãn để đếm số lần đổi lớp. Chạy cùng can thiệp trên điểm tuyến tính $Bl$ để có đối chứng trên đúng các sketch.

| Số sketch giữ hạng nhất sau mọi phép bỏ một từ | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin |
|---|---:|---:|---:|
| Ảnh đầu giữ nguyên, chấm tuyến tính | 8 / 100 | 11 / 84 | 10 / 120 |
| Ảnh đầu giữ nguyên, xác nhận theo từ | 7 / 100 | 8 / 84 | 11 / 120 |
| Lớp của ảnh đầu giữ nguyên, chấm tuyến tính | 79 / 100 | 64 / 84 | 98 / 120 |
| Lớp của ảnh đầu giữ nguyên, xác nhận theo từ | 82 / 100 | 65 / 84 | 101 / 120 |

Trong 304 sketch, 26 giữ nguyên ảnh đầu với mọi phép bỏ từng từ, còn 248 giữ nguyên lớp. Đối chứng tuyến tính có 29 và 241. Trên các sketch ghép cặp, 13 chuyển từ có thể đổi lớp sang giữ nguyên lớp và 6 chuyển chiều ngược lại. Tính ổn định này xét theo lớp ảnh đang được dự đoán, kể cả lúc dự đoán ban đầu sai. Cận sàng lọc loại được 5.782.009 trên 6.363.024 cặp sketch–từ (90,87%). Một sketch ấn định trước ở mỗi split còn được vét cạn thực tế cả 20.931 phép xóa; số từ bị sàng lọc nhầm làm đổi ảnh đầu là 0 ở cả ba. Sai khác lớn nhất quan sát khi đối chiếu điểm sau can thiệp trong phần toàn gallery là 6,67 × 10⁻⁶ theo điểm hạng nhất ban đầu.

Trên TU-Berlin, một sketch lớp `helicopter` trả ảnh `mosquito` đầu tiên; khi bỏ tín hiệu của từ `mosquito`, ảnh `helicopter` lên đầu. Trên Sketchy Ext-1, một sketch lớp `cup` bị xếp vào `snail`; bỏ từ `snail` chuyển kết quả sang `teddy_bear`, vẫn sai. Một sketch `cup` khác đổi sang một ảnh cốc khác khi bỏ `mug`. Các ví dụ lấy theo thứ tự trong mẫu đã ấn định; tệp kèm có đường dẫn ảnh và chênh lệch điểm trước/sau.

Phương pháp xếp hạng và công thức chẩn đoán có thể sử dụng cùng dữ liệu ảnh–từ cho các cặp miền khác. Thí nghiệm này chỉ đo sketch → photo. Chứng nhận áp dụng cho việc đặt một tọa độ likelihood của sketch bằng 0; thay đổi pixel, tên từ trong gallery, cả một nhóm từ và thay đổi nhiều tọa độ là các can thiệp khác. Kết quả mAP và độ bền của hạng nhất trả lời hai câu hỏi khác nhau. [Kết quả truy hồi](../gallery-lexical-feedback/README.md) · [Đối chứng ICFRR](../icfrr-word-control/README.md) · [Đối chứng tuyến tính](../word-deletion-linear-control) · [Ví dụ truy hồi](case-details.json) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU](tuberlin.json) · [Khóa](freeze.json) · [Từng truy vấn](per-query.zip)
