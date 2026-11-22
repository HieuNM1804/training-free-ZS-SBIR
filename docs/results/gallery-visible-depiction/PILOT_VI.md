# Phần chênh lệch giữa hai prompt có thể quan sát qua gallery

Với mỗi từ, sketch cho hai điểm CLIP theo cách mô tả “sketch” và “photo”. Phương pháp đo phần chênh lệch có thể suy ra lại từ sự thay đổi xác suất của các ảnh trong gallery. Hai từ dẫn đến cùng một phân bố ảnh có thể có điểm rất khác trong văn bản, nhưng phần chênh lệch đối lập đó triệt tiêu khi đi qua gallery. Cấu hình được chọn trên mẫu tăng phần quan sát được với hệ số 0,8 và giảm phần còn lại với hệ số −0,4 ở cả toàn ảnh lẫn các vùng. [Giao thức và công thức](../../GALLERY_VISIBLE_DEPICTION.md)

| mAP / precision (%) | Sketchy Ext-1 | Sketchy Ext-2 |
|---|---:|---:|
| Mốc vùng và từ | 84,64 / 83,96 | 80,63 / 78,38 |
| Tách thành hai phần qua gallery, áp dụng toàn ảnh và vùng | 84,43 / 83,69 | 81,18 / 79,26 |
| Chỉ áp dụng toàn ảnh | 84,52 / 83,80 | 81,02 / 79,01 |
| Chỉ áp dụng các vùng | 84,56 / 83,80 | 80,82 / 78,67 |
| Giữ hệ số, thay phần quan sát được bằng một tỷ lệ của chênh lệch ban đầu | 84,74 / 84,03 | 80,12 / 77,90 |
| Dò cả hệ số riêng cho đối chứng tỷ lệ | 84,43 / 83,74 | 81,10 / 78,75 |
| Xáo liên kết từ và ảnh trong gallery, dò riêng | 84,50 / 83,81 | 81,05 / 78,78 |
| Ngoại suy prompt trực tiếp, dò riêng | 84,40 / 83,65 | 81,09 / 78,87 |

Ở toàn ảnh, phần phương sai chuyển được sang các photo trung bình là 8,40% trên Ext-1 và 9,65% trên Ext-2. Với các vùng, số đo tương ứng là 42,65% và 47,65%. Hai chỉ mục có kích thước từ vựng khác nhau, nên số đo này chưa tách được tác dụng của vùng khỏi tác dụng giới hạn từ vựng. Tại Ext-2, phương pháp chính cao hơn 1,06 điểm mAP so với đối chứng tỷ lệ ở cùng hệ số, nhưng chỉ cao hơn 0,08 sau khi dò đối chứng riêng. So với nền, chênh lệch là −0,21 và +0,55 điểm mAP, với khoảng tin cậy theo lớp đều chứa 0. [Kết quả pilot và đối chứng](pilot.json)

Mỗi lớp đóng góp 36 sketch, tổng 900 và 756 truy vấn, dùng toàn bộ 17.100 và 12.552 ảnh. Các cấu hình cuối trong bảng có kiểm tra bằng KD-SBIR gốc trên Colab L4. [Các điểm từng sketch](pilot-per-query.zip) · [Ext-1](pilot-sketchy_1.json) · [Ext-2](pilot-sketchy_2.json)
