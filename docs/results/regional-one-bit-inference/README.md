# Một bit từ sketch có đủ để kiểm tra bằng chứng riêng của ứng viên?

**Lỗ hổng của phép thử trước.** Hai nửa từ vựng chứa 10.466 và 10.465 từ. Sau khi trừ trung bình, cả hai ma trận text embedding đều có hạng **512/512**. Khi biết đủ các log odds trên một nửa, có thể giải ra vector biến đổi của sketch bằng đại số tuyến tính. Trên các ca kiểm tra ở hai split, sai số logit lớn nhất ở nửa giữ riêng nhỏ hơn 1,5×10⁻¹¹ khi kiểm tra đẳng thức trong float64. Như vậy việc dự đoán các từ giữ riêng thành công trong [phép thử nhiều từ](../regional-conditional-repertoire/README.md) tự nó không chứng minh ảnh bổ sung thông tin. Mỗi hàng logit chỉ là một phép chiếu khác của cùng vector.

**Phép thử sửa lại.** Chỉ cho biết ghép hai vùng đã làm tăng hay giảm log odds của từ u so với từ v. Yêu cầu dự báo hướng thay đổi cho cặp từ s,t. Một ảnh đề xuất tập các hướng ngữ nghĩa của sáu tương tác vùng. Xác suất hai dấu giống nhau được tính giải tích từ góc giữa hai cặp từ sau khi chiếu vào tập hướng ấy. Chọn 1.024 bộ bốn từ cho mỗi cặp vùng và chấm các bộ có bốn từ khác nhau; tỷ lệ giữ lại là 92,74% / 93,25% khi lấy mẫu theo sketch trước khi ghép vùng. Các đối chứng nhìn thấy cùng một bit.

| Giữa hai ảnh cùng lớp, sketch ban đầu, ngữ cảnh trung tâm | Ext-1 | Ext-2 |
|---|---:|---:|
| Brier loss khi chỉ dùng hình học từ CLIP phía sketch | **0,24526** | **0,24565** |
| Brier khi luôn cho hai kết quả xác suất 1/2 | 0,25000 | 0,25000 |
| Brier dùng tập hướng tương tác của đúng ảnh mẫu | 0,26424 | 0,26454 |
| Mức giảm Brier của tương tác so với các hướng cộng tính của đúng ảnh | +0,01061 | +0,01076 |
| Chọn đúng ảnh nguồn so với đối thủ | 47,67% | 56,35% |

Brier càng thấp thì xác suất dự báo càng tốt. Tập hướng tương tác dự báo tốt hơn tập hướng cộng tính của chính ảnh nhưng vẫn kém cả dự báo 1/2. Lợi ích phân biệt đúng trừ sai so với dùng hướng cộng tính là −0,00045 / +0,00008, khoảng bootstrap theo cặp chứa 0 trên cả hai split. Với đối thủ khác lớp, thay đổi biên là −0,00165 / −0,00124 với các khoảng âm. Lợi ích quan sát được khi chỉ xét ảnh đúng cũng có thể xuất hiện ở ảnh sai. Trong thiết kế này chưa có bằng chứng đủ để sử dụng tập hướng ấy như một cơ chế cải thiện truy hồi. [Protocol, công thức và nguồn nghiên cứu](protocol.md) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [Số liệu từng truy vấn](paired-T4.zip)
