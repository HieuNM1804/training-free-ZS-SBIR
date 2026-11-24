# Một ứng viên có dự báo đúng phần chưa thấy của sketch?

Cho mô hình xem một crop sketch và một ảnh ứng viên. Từ đó dự đoán phân bố từ CLIP tại một crop khác, không chồng pixel. Ảnh thật ban đầu được mô tả bằng token của toàn ảnh, còn từng vùng sketch được đưa riêng qua encoder. Phép thử dùng các cặp ảnh gốc cùng lớp có hình thức gần nhau. Ext-1 có 3.379 sketch và 300 cặp, Ext-2 có 2.990 sketch và 252 cặp. ∆ log score tính bằng nat, giá trị cao hơn là tốt hơn.

| Đo tác động của bố cục thật so với dịch sai | Ext-1 | Ext-2 |
|---|---:|---:|
| Cải thiện dự báo vùng chưa thấy bằng ảnh đúng | −0,00021 | +0,00145 |
| Cải thiện dự báo bằng ảnh đối thủ | −0,00009 | +0,00127 |
| Chênh lệch hai tác động | −0,00012 | +0,00018 |

Trên Ext-2, bố cục thật giúp dự báo crop thứ hai ở cả ảnh đúng và đối thủ. Khoảng bootstrap 95% cho lợi ích phân biệt ảnh đúng với đối thủ chứa 0 trên cả hai split. Khi điểm toàn ảnh ghép sai hai nguồn, phần tăng thêm của dự báo vùng ghép đúng 50,95% / 51,88%. Khoảng 95% tương ứng là [47,62%, 54,27%] và [48,48%, 55,46%]. Phép đo ghép hai nguồn triệt tiêu xu hướng cố định thích một ảnh hơn ảnh kia. Nó chỉ được tính khi đánh giá sau khi từng sketch đã nhận điểm riêng.

**Đối chứng dương phát hiện giới hạn của phép đo.** Giữ nguyên ảnh, vùng, prompt, từ và các cặp đánh giá. Khi ảnh gallery cũng được cắt ra và đưa độc lập qua CLIP, dự báo cùng tọa độ ghép đúng nguồn **69,75% / 66,69%**. Dùng token toàn ảnh cho cùng phép đọc chỉ đạt **41,31% / 42,63%**. Với ảnh truy vấn chính là ảnh tham chiếu, hai kết quả là 100% / 100% và 36,67% / 39,29%. Thí nghiệm thay đồng thời trường tiếp nhận, tỉ lệ phóng crop và cách gộp token, nên chưa tách nguyên nhân nào chi phối. Đây là chẩn đoán fine-grained dựa trên nguồn ảnh thật của Sketchy.

[Thiết kế ban đầu](protocol.md) · [Phân tích loại thiên lệch cặp ảnh](source-assignment.json) · [Thiết kế đối chứng dương](observation-control.md) · [Kết quả đọc vùng theo hai cách](../regional-observation-control/README.md) · [Truy vấn và số liệu gốc](paired-T4.zip)
