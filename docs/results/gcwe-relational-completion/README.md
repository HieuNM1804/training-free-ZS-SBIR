# Hai ảnh khác nhau có thể giải thích hai quan hệ của một bộ phận còn thiếu?

Lấy hai vùng sketch đã thấy A,B và một vùng C cần dự báo. Mỗi chữ c là một giả thuyết về ý nghĩa cục bộ của C, đo bằng frozen CLIP trên crop. Có thể dùng một ảnh cho cả hai quan hệ A–C và B–C, hoặc lấy hai ảnh khác nhau có chung c. Nhánh thứ hai tính chính xác hậu nghiệm khi hai ảnh phải khác ID. Mọi mô hình dùng cùng panel ảnh và cùng quan sát. Đối chứng bỏ ràng buộc lân cận dùng tất cả 24 hoán vị thay cho tám phép đối xứng của hình vuông.

| Log score vùng C, mô hình hai ảnh khác nhau trừ đối chứng (nat) | Ext-1 | Ext-2 |
|---|---:|---:|
| Một ảnh và cùng hệ tọa độ cho cả ba vùng | +0,00118 | +0,00135 |
| Hai ảnh được rút độc lập, có thể trùng ID | +0,00148 | +0,00169 |
| Bỏ quan hệ lân cận hình học | +0,00026 | −0,00050 |
| Phân bố từ trước bước cập nhật, trên cùng panel | −0,00793 | −0,00634 |

Mức tăng nhỏ so với dùng một ảnh có khoảng bootstrap 95% theo lớp chứa 0 trên cả hai split. Khi bỏ một phần tư ảnh có hỗ trợ mạnh nhất cho cả hai vùng, mức chênh so với một ảnh là +0,00008 / −0,00099 nat. Mô hình có dạng xác suất rõ ràng và vượt qua kiểm tra tổng hữu hạn trực tiếp. Bằng chứng vẫn chưa cho thấy việc ghép quan hệ của hai ảnh phục hồi được ý nghĩa còn thiếu tốt hơn phân bố đã có.

Trên 768 sketch mới mỗi split và gallery đầy đủ, truy hồi chỉ bằng phần ngữ nghĩa suy từ các crop đạt **12,141 / 5,370** và **8,275 / 8,098** (mAP / precision). Đường suy qua một ảnh trực tiếp đạt **14,846 / 8,503** và **11,138 / 9,971**. Hạng tử ngữ nghĩa toàn ảnh gốc đạt **80,463 / 79,521** và **74,711 / 73,283** trên cùng truy vấn. Các số này là kết quả của nhánh đứng riêng. [Mô hình, nguồn và tiêu chí](protocol.md) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [Kết quả theo truy vấn](paired-T4.zip)
