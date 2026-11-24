# Giữ nghĩa toàn ảnh cố định để đo riêng sự phụ thuộc giữa các từ

Trung bình các phân bố từ của 16 ảnh thiếu vùng có thể khác phân bố từ của ảnh nguyên vẹn. Do đó lợi ích của việc dùng các ảnh thiếu vùng có thể đến từ thay đổi nội dung ngay cả trước khi nhìn sketch. Trên ảnh nguồn của các cặp cùng lớp, KL từ phân bố ảnh nguyên vẹn tới trung bình ban đầu là 0,1953 / 0,1629 nat. Để tách yếu tố đó, tạo một phân bố chung trên từ và tập vùng. Phép chiếu KL giữ nguyên xác suất của tập vùng khi đã biết từ, đồng thời buộc phân bố biên của từng từ bằng đúng phân bố của ảnh nguyên vẹn. Mọi giả thuyết khi chỉ quan sát một từ vì thế có cùng likelihood. Chúng khác nhau ở dự báo về từ thứ hai.

| Dự báo bản vẽ độc lập, mô hình chung tập vùng trừ đối chứng (nat) | Ext-1 | Ext-2 |
|---|---:|---:|
| Ảnh đúng, giữ số vùng nhưng chọn lại vùng (đối thủ cùng lớp) | +0,06775 | +0,06538 |
| Ảnh đúng, tráo danh tính tập vùng riêng cho từng từ (đối thủ cùng lớp) | +0,05772 | +0,05817 |
| Biên đúng trừ sai so với giữ số vùng (đối thủ cùng lớp) | +0,00074 | −0,00600 |
| Biên đúng trừ sai so với giữ số vùng (đối thủ khác lớp) | **−0,08174** | **−0,07132** |
| Biên đúng trừ sai so với tráo danh tính tập vùng (đối thủ khác lớp) | −0,07967 | −0,06910 |

Việc hai từ cùng ưu tiên một phần ảnh có ích để dự báo từ của bản vẽ khác. Khi so hai ảnh khác lớp, mô hình lại giúp đối thủ sử dụng từ thứ nhất để giải thích từ thứ hai nhiều hơn ảnh thật. Khoảng 95% theo cặp cho hàng giữ số vùng là [−0,10565, −0,05971] / [−0,09437, −0,05059]. Đưa cả xác suất của từ đầu vào điểm chung thay đổi độ chính xác tuyệt đối, nhưng theo đẳng thức về phân bố biên nó không thay đổi chênh log score giữa các phương pháp trên cùng ảnh. Độ chính xác chọn nguồn theo điểm chung là 52,67% / 56,35% giữa hai ảnh cùng lớp và 89,60% / 83,73% khi hai ảnh khác lớp. Những số này dùng ảnh nguồn, chưa phải truy hồi theo lớp trên toàn gallery.

Sự thất bại trong phân biệt ảnh vẫn xuất hiện sau khi đã loại khả năng do sai lệch phân bố biên. Một giả định còn lại là coi từ rút từ sketch như một nhận diện chắc chắn mà bất kỳ ảnh nào cũng phải giải thích. [Phép thử tiếp theo với bằng chứng từ chưa chắc chắn](../semantic-common-depiction/protocol.md) · [Suy dẫn, giả thuyết và prior art](protocol.md) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [Dữ liệu từng cặp](paired-T4.zip)
