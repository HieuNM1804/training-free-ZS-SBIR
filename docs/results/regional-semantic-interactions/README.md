# Ảnh ứng viên có dự báo được nghĩa xuất hiện khi ghép hai vùng sketch?

Cho CLIP đọc một vùng trung tâm C, rồi đọc C+A và C+B riêng biệt. Từ ba lần đọc đó, dự đoán phân bố 20.931 từ nếu hai phần A và B cùng xuất hiện. Hiệu giữa lần đọc thực C+A+B và dự báo cộng tính là phần ngữ nghĩa nảy sinh khi hai vùng có mặt cùng nhau. Lấy phần tương ứng trên ảnh ứng viên để dự báo cho sketch. Cách này tạo một mục tiêu chưa được nhìn thấy, đồng thời khử ảnh hưởng của việc chuẩn hóa vector đặc trưng. CLIP chỉ nhận các patch đã chọn với vị trí gốc. Khi đưa đủ patch vào, phép tính tái lập đúng embedding chuẩn trong batch kiểm tra của cả ảnh và sketch.

| Giữa hai ảnh nguồn cùng lớp, ngữ cảnh trung tâm | Sketchy Ext-1 | Sketchy Ext-2 |
|---|---:|---:|
| Cặp ảnh nguồn / quyết định trên sketch | 150 / 300 | 126 / 252 |
| Dư địa tối đa để cải thiện dự báo cộng tính (nat) | 0,1774 | 0,1502 |
| Thay đổi log score của ảnh nguồn khi thêm tương tác từ chính ảnh (nat) | −0,1775 | −0,1487 |
| Chọn đúng ảnh nguồn cho từng sketch | 49,00% | 48,41% |
| Thay đổi biên đúng trừ sai so với dùng hai vùng đối diện trong cùng ảnh (nat) | +0,00285 | +0,00043 |

Khoảng bootstrap theo cặp ảnh của hàng cuối là [−0,00425, 0,01036] và [−0,00607, 0,00666]. Ở ngữ cảnh thứ hai, khi giữ cả hai vùng còn lại trong sketch, dư địa chỉ còn 0,0834 / 0,0674 nat. Chuyển tương tác thật của ảnh vào dự báo vẫn làm mất 0,0560 / 0,0494 nat. Như vậy CLIP có phản hồi phi cộng tính rõ ràng, nhưng chuyển nguyên độ lớn và hướng phản hồi đó qua hai miền chưa cải thiện dự báo. Mỗi đối thủ cùng lớp cũng là đáp án đúng khi chấm SBIR theo lớp. Các tỷ lệ trên là độ chính xác nhận diện ảnh nguồn, không phải mAP. [Thiết kế, công thức và nguồn](protocol.md) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [Điểm theo từng ca](paired-T4.zip)
