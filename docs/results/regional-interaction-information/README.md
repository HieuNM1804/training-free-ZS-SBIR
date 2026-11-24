# Phần tương tác thất bại ở hướng ngữ nghĩa hay ở mức tự tin?

Với phân bố đích t, dự báo cộng tính b và thay đổi logit d do ảnh đề xuất, log score tăng đúng bằng A−C, với A=(t−b)·d và C=KL(b || softmax(log b+d)). A đo hướng thay đổi từ có khớp sai số dự báo trên sketch hay không. C là giá phải trả khi chuyển khối lượng xác suất. Mã kiểm tra đẳng thức với cách tính log score trực tiếp và tái lập điểm của phép thử trước trên từng ca. Các số dưới đây lấy trung bình trên những ảnh đúng trong nhóm đối thủ cùng lớp, cùng ngữ cảnh trung tâm.

| Nguồn dự đoán tương tác | Hướng A, Ext-1 / Ext-2 | Chi phí C, Ext-1 / Ext-2 | Lợi ích A−C, Ext-1 / Ext-2 (nat) |
|---|---:|---:|---:|
| Ảnh thật đã được dùng làm mẫu vẽ | +0,00307 / +0,00208 | 0,18059 / 0,15082 | −0,17752 / −0,14875 |
| Bản vẽ khác của cùng ảnh, chỉ dùng để chẩn đoán | +0,03372 / +0,03404 | 0,15712 / 0,12939 | −0,12339 / −0,09536 |

Ngay cả khi ảnh dự báo đúng hướng ở ước lượng trung bình, nó có thể làm log score giảm mạnh. Khoảng 95% theo cặp cho A của đúng ảnh thật là [−0,00434, 0,01023] / [−0,00384, 0,00827]. Thay ảnh thật bằng một bản vẽ khác cung cấp hướng sát sketch hơn, nhưng đối thủ cùng lớp cũng chia sẻ một phần tín hiệu. Trong ngữ cảnh đủ các vùng còn lại, chênh lệch hướng đúng trừ sai của bản vẽ lại là +0,00775 / +0,00572 nat, khoảng theo cặp dương trên cả hai split; với ảnh thật là +0,00300 / +0,00122 và cả hai khoảng chứa 0. Có bằng chứng một phần nghĩa khi ghép vùng lặp lại giữa các bản vẽ. Suy ra nó từ ảnh thật vẫn là vấn đề chưa giải quyết. Các bản vẽ cùng nguồn chỉ được ghép sau khi xác định truy vấn, để đo tính lặp lại. [Protocol và giới hạn diễn giải theo hệ số](protocol.md) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [Số liệu ghép cặp](paired-T4.zip)
