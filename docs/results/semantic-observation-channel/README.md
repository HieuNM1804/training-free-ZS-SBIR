# Một cách kiểm tra thông tin bị lược khi chuyển từ ảnh sang sketch

Giả thuyết là mỗi từ có một mức dễ được thể hiện trong sketch, dùng chung giữa các vùng của cùng truy vấn. Với một ảnh ứng viên, ước lượng mức đó từ ba crop đã thấy rồi dự đoán crop thứ tư bằng phân bố từ CLIP. Trong dạng log, dự đoán trước chuẩn hóa bằng **trung bình log từ của sketch ở ba vùng** cộng **chênh log từ của ảnh ứng viên ở vùng đích so với ba vùng kia**. Vì thế mức từ tuyệt đối của ảnh bị triệt tiêu. Phép thử có thể xác định khả năng chuyển tương phản giữa các vùng; để suy ra xác suất con người lược một thuộc tính cần thêm quan sát độc lập.

| Ảnh nguồn cùng lớp · mô hình hiệu chỉnh trừ đối chứng, nat | Ext-1 | Ext-2 |
|---|---:|---:|
| Log score dự báo bằng ảnh đúng so với lấy từ ảnh nguyên bản | +1,3095 | +1,2579 |
| Biên ảnh đúng trừ ảnh sai so với lấy từ ảnh nguyên bản | +0,0722 | +0,0562 |
| Biên ảnh đúng trừ ảnh sai so với gán mức hiệu chỉnh của ứng viên kia | +0,0725 | +0,0834 |
| Log score dự báo bằng ảnh đúng so với chỉ đọc ba vùng của sketch | −1,0562 | −1,0577 |

Khoảng bootstrap theo cặp cho lợi ích biên so với dùng ảnh nguyên bản là [0,0346; 0,1132] và [0,0056; 0,1027] nat. Trên 1.676 / 1.528 sketch ở 150 / 126 cặp ảnh cùng lớp, độ chính xác chọn nguồn cho một sketch tăng từ 53,46% / 54,52% lên 57,76% / 57,92%. Độ chính xác của phép ghép hai nguồn lại giảm từ 68,65% / 70,29% xuống 67,63% / 69,34%. Với đối thủ khác lớp, biên ảnh đúng trừ ảnh sai giảm 0,1100 / 0,1795 nat dù log score của ảnh đúng vẫn tăng mạnh. Cải thiện dự báo tuyệt đối không đủ chứng minh cải thiện truy hồi theo lớp.

Nhánh suy hướng từ tỉ số odds ghép hai từ đúng 52,38% / 52,34% khi tính trung bình trên cặp cùng lớp. Khi bỏ cấu trúc vị trí, trung bình trên đủ 24 hoán vị tạo xác suất 1/2 theo đối xứng. Ở lượt đầu, sai khác cỡ làm tròn quanh 1/2 đã tạo thứ hạng biểu kiến cho đối chứng này. Mã chấm đã sửa bằng cách kiểm tra đẳng thức rồi gán chính xác 1/2, tính hòa là nửa điểm. [Thiết kế, toán học và bằng chứng](protocol.md) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [Dữ liệu ghép cặp](paired-T4.zip)
