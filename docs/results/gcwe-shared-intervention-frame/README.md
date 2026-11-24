# Giữ cùng hệ tọa độ khi yêu cầu hai ảnh giải thích một sketch

Một ảnh có tương ứng patch rõ hơn thường chọn hệ tọa độ quyết đoán hơn. Nếu dự báo từ của nó chỉ hơi tốt hơn đối thủ, việc quá tự tin có thể khiến nó thua khi chấm xác suất. Trên cùng 150 / 126 cặp nguồn cùng lớp và 149 / 126 cặp khác lớp, đổi chéo ảnh dùng để suy ánh xạ và ảnh cung cấp phản hồi từ. Chấm cả bốn tổ hợp. Sau đó cho hai ứng viên dùng chung trung bình của hai phân bố ánh xạ; phép lấy trung bình diễn ra trước khi chấm dự báo.

| Sketch nguyên gốc | Ext-1, đối thủ cùng lớp | Ext-2, đối thủ cùng lớp | Ext-1, đối thủ khác lớp | Ext-2, đối thủ khác lớp |
|---|---:|---:|---:|---:|
| Biên Brier thuận cho ảnh đúng, mỗi ảnh tự chọn hệ tọa độ | −0,00016 | −0,00146 | −0,00120 | +0,00008 |
| Biên Brier thuận cho ảnh đúng, cùng phân bố tọa độ | +0,00037 | +0,00079 | +0,00282 | +0,00196 |
| Chọn đúng nguồn, mỗi ảnh tự chọn tọa độ | 49,33% | 46,03% | 46,31% | 49,21% |
| Chọn đúng nguồn, cùng phân bố tọa độ | 50,00% | 50,79% | 54,70% | 54,37% |

Trong nhóm khác lớp, khoảng bootstrap theo cặp của biên dùng chung tọa độ là [0,00090, 0,00486] và [0,00022, 0,00371]. Độ chính xác nguồn tăng 8,39 / 5,16 điểm phần trăm, nhưng khoảng của mức tăng ở Ext-2 vẫn chứa 0. Trong nhóm cùng lớp, cả biên lẫn thay đổi độ chính xác đều chưa tách được khỏi 0. Khi lấy riêng hiệu ứng của việc đổi ảnh cung cấp từ và giữ cố định phân bố hình học, biên cùng lớp là +0,00021 / +0,00093, hai khoảng đều chứa 0. Tiêu chí ban đầu để khẳng định hiệu ứng ngữ nghĩa riêng của từng ảnh chưa đạt.

Trên đúng ảnh nguồn ở nhóm cùng lớp, hình học cho phép xác suất gán từ lệch khỏi 1/2 tối đa trung bình 0,310 / 0,293 theo chặn đối xứng. Mức lệch tuyệt đối thực tế là 0,091 / 0,083; mức lệch có dấu về phía đáp án đúng chỉ 0,0072 / 0,0046. Như vậy phân bố hình học đã phá được một phần đối xứng, nhưng chiều của bằng chứng từ thường không trùng với chiều cần dự báo. Kiểm tra ảnh cùng lớp khác nguồn ở biên truy hồi là cần thiết trước khi dùng kết quả giữa hai lớp làm bằng chứng cho SBIR. [Protocol và đẳng thức tách hai hiệu ứng](protocol.md) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [Kết quả ghép cặp](paired-T4.zip)
