# Hai ảnh cùng lớp có thể chọn vùng thay cho nhau không?

Lập hai danh sách ảnh chọn vùng cho mỗi ứng viên. Một danh sách gồm ảnh khác cùng lớp, danh sách kia gồm ảnh khác lớp. Mỗi cặp ảnh chọn vùng cách ứng viên đúng bằng nhau về hạng toàn sketch, và mỗi nhóm đóng góp một ảnh phía trên cùng một ảnh phía dưới ứng viên. Hai ảnh đang so sánh không được làm ảnh chọn vùng cho nhau. Điểm toàn ảnh, điểm từng crop và trọng số cho từng ảnh đều được tạo độc lập với nhãn. Việc dùng nhãn để đối chiếu danh tính lớp tạo một phép chẩn đoán có giám sát hậu nghiệm. [Thiết kế can thiệp](protocol.md)

| Ảnh chọn vùng cùng lớp trừ khác lớp | Ext-1 | Ext-2 |
|---|---:|---:|
| Cặp có đủ ảnh đối chứng / cặp sát hạng khác tính liên quan | 192 / 878 | 1.205 / 2.105 |
| Số sketch có cặp đạt điều kiện / có đủ hai hướng | 30 / 18 | 131 / 97 |
| Thay đổi log score của ảnh đúng | +0,4449 | +0,1228 |
| Thay đổi log score của ảnh sai | +0,1352 | +0,0658 |
| Thay đổi biên đúng trừ sai, trung bình theo sketch | +0,3097 | +0,0569 |
| Thay đổi độ chính xác cân bằng theo sketch (điểm %) | +1,65 | +0,70 |

Khoảng bootstrap 95% theo lớp của chênh lệch biên là [0,1533, 0,9186] / [−0,0319, 0,3632]. Khoảng tương ứng cho thay đổi độ chính xác cân bằng là [0,00, 5,13] / [−4,52, 20,51] điểm. Hai estimand dùng trọng số khác nhau: bảng cho trọng số bằng nhau cho sketch đủ điều kiện, còn bootstrap cho trọng số bằng nhau cho lớp đủ điều kiện. Việc ảnh cùng lớp tăng điểm cho một ứng viên không chứng minh rằng chúng giúp đặt ảnh đúng lên trên đối thủ. [Kết quả Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [ID ảnh chọn vùng và điểm ghép cặp](paired-T4.zip) · [Xác nhận trên nguồn ảnh độc lập](../regional-selector-independent-sources/README.md)
