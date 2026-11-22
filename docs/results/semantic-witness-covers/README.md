# Tập ảnh cùng giải thích các từ do một sketch gợi ra

Pilot gồm 900 và 756 sketch, 36 sketch mỗi lớp trên hai split, với toàn bộ 17.100 và 12.552 ảnh gallery. Các con số là mAP / precision (%) từ evaluator KD-SBIR trên Colab T4. Hai split unseen đều tham gia chọn cấu hình thăm dò.

| Phương án | Ext-1 | Ext-2 |
|---|---:|---:|
| Nền vùng | 84,64 / 83,94 | 80,64 / 78,38 |
| Phủ 8 từ bằng 4 ảnh | 84,72 / 83,82 | 80,58 / 78,12 |
| Tráo các từ trong từng cụm, tự chọn cấu hình | 84,67 / 83,82 | 80,60 / 78,20 |
| Dùng một ảnh, tự chọn cấu hình | 84,62 / 84,09 | 80,47 / 77,99 |
| Chỉ giữ phần phụ thuộc giữa các từ | 84,58 / 84,05 | 80,54 / 78,29 |

Cấu hình chính gom gallery thành 24 cụm ảnh, lấy 8 từ, ngưỡng phân vị 90%, 4 lượt lấy ảnh độc lập và hệ số điểm 0,1. Tráo từng từ giữ đúng tỷ lệ xuất hiện của từ đó trong từng cụm. Phương pháp thật hơn phương án tráo 0,05 điểm mAP trên Ext-1 và thấp hơn 0,01 trên Ext-2. Chưa có bằng chứng rằng quan hệ bổ sung giữa ảnh, sau khi biết tần suất từng từ, cải thiện truy hồi. Lượt này dừng ở pilot hai split và chưa chấm trên TU-Berlin.

Các tệp đi kèm lưu lưới tham số, từng truy vấn, đối chứng khớp tham số và các lần phát lại evaluator. [Cơ chế phủ tập, nguồn paper và đối chứng](../../SEMANTIC_WITNESS_COVERS.md)
