# Ước lượng phần điểm truy hồi do một quan sát thiếu nội dung tạo ra

Trên pilot L4 của nhánh query-fragment-reference, thay sketch bằng vector của ảnh đầy đủ trừ 0,8 lần vector trung bình sáu mảnh rồi truyền điểm trên gallery đạt mAP 74,89 và 74,44. Một ảnh trắng cố định cũng giải thích phần lớn mức tăng so với cosine CLIP. Hai kết quả đặt ra câu hỏi liệu có thể ước lượng hệ số trừ từ ảnh đang hỏi và gallery chưa gán nhãn, với cùng một công thức cho mọi truy vấn.

## Giả thuyết và phép đo

Gọi q là embedding chuẩn hóa của truy vấn đầy đủ, f là trung bình embedding CLIP của sáu quan sát từng phần và pᵢ là embedding của ảnh gallery. Nếu hai ảnh đích gần nhau về nội dung, chênh lệch giữa điểm q·pᵢ và q·pⱼ có thể mang tín hiệu do cách thể hiện hoặc bề mặt dễ gây nhiễu. Xét các cặp ảnh có trọng số wᵢⱼ lấy từ chính đồ thị RGB không nhãn. Đặt C bằng trung bình có trọng số của (pᵢ−pⱼ)(pᵢ−pⱼ)ᵀ. Hệ số bình phương tối thiểu β=(qᵀCf)/(fᵀCf) giải thích phần chênh lệch điểm dự đoán được từ quan sát thiếu. Chấm với duy nhất vector q−βf. Hệ số được tính riêng cho từng truy vấn. Phép chiếu có dạng đóng và giữ được chỉ mục vector của gallery. Có thể đọc điểm trực tiếp hoặc truyền điểm đã sửa qua cùng đồ thị với mọi đối chứng.

Điều kiện nhận dạng quan trọng là các thay đổi ngữ nghĩa còn lại giữa hai ảnh kề nhau ít tương quan với phản ứng của f. Hai ảnh gần nhau vẫn có thể khác lớp. Tín hiệu có ích của chính đối tượng cũng có thể sống sót qua việc chỉ cho thấy một mảnh. Do đó tính trực giao trên những cạnh dùng để tính β chỉ xác nhận cài đặt. Khả năng cứu kết quả sai và chuyển giao mới kiểm tra được giả thuyết.

## Quyết định thực nghiệm trước khi đo

Cấu hình chính dùng nhóm patch theo token gốc, hệ số từ các cạnh lân cận thực và cùng toán tử RGB. Hệ số trừ được xác định hoàn toàn từ ảnh, nhiệt độ đồ thị 64 và tham số đồ thị được kế thừa từ pilot Sketchy trước. Các đối chứng dùng cùng ảnh, compute đọc điểm và nhiệt độ. Thay metric cạnh thật bằng hiệp phương sai toàn gallery, khoảng cách Euclid, hoặc cùng đồ thị gán lại ngẫu nhiên cho các ảnh. Thay hệ số tự tính bằng 0,8 như thí nghiệm trước. Thay các mảnh bằng lưới, các vùng ngẫu nhiên cùng kích thước, mảnh xáo bố cục, token trung bình, ảnh màu trung bình, ảnh trắng hoặc ảnh trung tính. Đo cả kết quả trước và sau propagation. Ảnh trắng là đối chứng can thiệp đặc biệt phù hợp với sketch. Công thức đề xuất lấy mảnh từ chính ảnh và dùng cùng thuật toán cho ảnh có nền bất kỳ.

Đo mAP và precision, hệ số theo truy vấn, tương quan trước/sau trên các cạnh và tỷ lệ sửa cặp đúng–sai đã chọn bằng thứ hạng cosine gốc. Kiểm tra độc lập trên các ảnh gallery bằng cách chia đôi gallery bằng hoán vị cố định rồi ước lượng độ ổn định của β. Ghi cả kết quả âm khi hệ số từ cạnh thực kém hệ số cố định, cạnh tráo hoặc covariance toàn gallery. Chọn mọi hệ số phát sinh chỉ trên Sketchy và lưu khóa trước khi có điểm TU-Berlin.

## Công trình liên quan

Visual contrastive decoding dùng ảnh làm suy giảm để phát hiện phần dự đoán còn tồn tại khi thiếu thông tin ở bài toán sinh câu. LEACE nghiên cứu phép chiếu bỏ tín hiệu tuyến tính trong một metric hiệp phương sai. Diffusion trên lân cận ảnh cung cấp đồ thị không nhãn. Lựa chọn thay đổi cần kiểm tra ở đây là dùng **sai phân trên các cặp ảnh gallery để ước lượng lượng tín hiệu của từng truy vấn cần trừ**, rồi tái sử dụng đúng vector đó cho toàn gallery. [Leng và cộng sự, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Leng_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_through_Visual_Contrastive_CVPR_2024_paper.html) · [Belrose và cộng sự, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d066d21c619d0a78c5b557fa3291a8f4-Abstract-Conference.html) · [Wang và cộng sự, CVPR 2012](https://pages.ucsd.edu/~ztu/publication/cvpr12_CrDP.pdf)

QuARI đã công bố retrieval với phép chiếu được điều kiện theo truy vấn và học bằng hypernetwork. Điều này giới hạn mọi tuyên bố novelty rộng về phép chiếu theo truy vấn. [Xing và cộng sự, NeurIPS 2025](https://ericx003.github.io/proj/quari/)
