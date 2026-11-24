# Một nghĩa chung có còn giữ cùng vai trò không gian?

Hai ảnh được xem là cùng nhắc tới một từ với xác suất bằng tích vô hướng giữa hai phân bố từ toàn ảnh. Khi hai từ được cả sketch và ảnh gợi ý, cơ chế mới còn yêu cầu thứ tự gán hai từ vào hai vị trí phải nhất quán giữa hai miền. Odds của phép gán này bất biến khi nhân phân bố bằng hệ số riêng cho mỗi từ hoặc mỗi vị trí. Chứng minh và mã bao gồm đối chứng vẽ hai danh tính từ độc lập, xoay vị trí và hoán vị tên từ. Phép so sánh toàn ảnh vẫn dùng đủ 20.931 từ. Mỗi ảnh ứng viên chỉ có thông tin vị trí cho 32 từ đứng đầu của chính nó.

| 384 sketch mỗi split, toàn gallery | Ext-1 mAP@all / P@100 | Ext-2 mAP@200 / P@200 |
|---|---:|---:|
| Từ toàn ảnh | 76,058 / 79,630 | 72,143 / 71,569 |
| Cùng từ và cùng vai trò | 76,053 / 79,620 | 72,145 / 71,570 |
| Từ toàn ảnh sau diffusion RGB | 82,245 / 81,500 | 76,224 / 74,763 |
| Cùng từ và cùng vai trò sau diffusion RGB | 82,238 / 81,500 | 76,266 / 74,745 |

So với đối chứng toàn ảnh, thay đổi mAP sau diffusion là −0,007 / +0,042 điểm. Trong top 200 của điểm ngữ nghĩa gốc, xác suất cả hai từ nằm trong tập hỗ trợ vị trí trung bình đạt 0,908 / 0,870. Sau khi bỏ hai lần rút trùng từ, khối lượng còn 0,465 / 0,516. Chặn trên Cauchy–Schwarz của độ lớn tương quan vai trò trung bình là 0,0779 / 0,0797, còn độ lớn thực đo chỉ là 0,0204 / 0,0202. Hệ số nhân vào điểm toàn ảnh trung bình là 1,000419 / 1,000086. Hai chiều của bằng chứng vị trí thường triệt tiêu nhau ở mức tổng hợp. Các số này phân biệt giới hạn do danh tính từ bị trùng, mức xác định được vai trò ở từng miền và sự nhất quán giữa hai miền. Chưa đủ bằng chứng để giữ can thiệp này trong mô hình truy hồi. [Thiết kế và prior art](protocol.md) · [Suy dẫn bằng số](identifiability.json) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [Dữ liệu theo truy vấn](paired-T4.zip)
