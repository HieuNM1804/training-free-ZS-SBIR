# Một phép thử trực tiếp cho giả thuyết sketch giữ lại một phần thông tin của ảnh

Với mỗi từ công khai, đọc phân bố CLIP của từ đó trên bốn vùng riêng biệt của ảnh và sketch. Nếu khoảng cách giữa hai miền chỉ do thông tin vị trí bị giản lược hoặc dịch chuyển, phải tồn tại *cùng một* ma trận chuyển bốn trạng thái mô phỏng được các quan sát của nhiều từ. Giải bài toán quy hoạch tuyến tính Le Cam trên 32 từ rồi dùng ma trận ấy dự đoán 32 từ giữ riêng. Đổi vai hai nửa và lấy trung bình. Một bản vẽ khác của cùng ảnh chỉ xuất hiện ở bước chấm. Phép thử không cần nhãn để suy luận cho một cặp và cho phép các vùng thay đổi vị trí.

| Chẩn đoán | Ext-1 | Ext-2 |
|---|---:|---:|
| Chọn đúng ảnh nguồn giữa hai ảnh cùng lớp, trên sketch ban đầu | 52,83% | 51,59% |
| Như trên, tráo danh tính từ ở bước dự đoán | 53,17% | 48,81% |
| Chọn đúng nguồn trên một bản vẽ độc lập, hai ảnh cùng lớp | 53,50% | 51,59% |
| Chọn đúng nguồn giữa hai ảnh khác lớp, trên sketch ban đầu | 53,36% | 56,55% |
| Chọn đúng nguồn trên bản vẽ độc lập, hai ảnh khác lớp | 51,68% | 53,77% |

Mỗi split có 150 / 126 cặp ảnh cùng lớp và 149 / 126 cặp khác lớp. Khi dự đoán từ giữ riêng trên sketch ban đầu, ảnh đúng giảm sai số total variation so với một dự báo chung cho mọi từ, trung bình 0,00970 / 0,00917 ở nhóm cùng lớp. Các khoảng 95% theo cặp đều dương. Trên bản vẽ độc lập, mức giảm trở thành −0,00185 / +0,00150 và cả hai khoảng đều chứa 0. Trong nhóm đối thủ cùng lớp, lợi ích của giữ đúng danh tính từ đối với biên ảnh đúng trừ ảnh sai chỉ là +0,00022 / +0,00010, các khoảng cũng chứa 0. Ở nhóm khác lớp trên Ext-2, danh tính từ mang thông tin trên cả hai bản vẽ. Hiện tượng đó chưa tái lập trên Ext-1.

Phân tích này dùng bốn vùng và 64 từ xếp cao theo phân bố từ của gallery. Nó kiểm tra một biến đổi giữa hai ảnh được tạo bởi cùng encoder. Darrin và cộng sự đã dùng so sánh các thí nghiệm thống kê để **xếp các encoder** theo khả năng mô phỏng biểu diễn của nhau. Bài toán ghép từng ảnh với từng sketch, tối ưu suy luận hữu hạn chính xác và việc giữ riêng danh tính từ cùng bản vẽ là các khác biệt của phép thử này. Chưa có bằng chứng nó đem lại một thuật toán truy hồi tốt hơn. [Darrin và cộng sự, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/7e6f445a74cdb71931aac64f1e3f49c9-Paper-Conference.pdf) · [Nguồn lý thuyết, protocol và đối chứng](protocol.md) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [Dữ liệu từng cặp](paired-T4.zip)
