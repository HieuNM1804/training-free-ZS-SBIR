# Đổi vai trò của ba tên công khai trong mô tả CLIP

Mỗi phép thử lấy ba tên từ các nhóm ảnh có khả năng gây nhầm lẫn. Với mỗi cặp tên, CLIP đọc hai câu có cùng từ nhưng đổi vai trò. Cộng ba hiệu quanh một vòng sẽ triệt tiêu chính xác mọi đóng góp có thể viết thành điểm riêng của từng từ. Tôi còn loại phần tương quan tuyến tính còn lại với ba điểm từ đơn trên gallery, rồi dùng hai nửa gallery để đánh giá độ ổn định của tín hiệu.

| mAP / precision (%) trên mẫu | Sketchy Ext-1 | Sketchy Ext-2 |
|---|---:|---:|
| Mốc chọn vùng theo ảnh ứng viên | 84,64 / 83,96 | 80,63 / 78,38 |
| Dùng chu trình đã kiểm tra với cú pháp chủ thể và nền | 84,61 / 83,92 | 80,74 / 78,52 |
| Chu trình với danh sách hai tên có vai trò đối xứng | 84,59 / 83,89 | 80,73 / 78,46 |
| Phần so sánh bắc cầu từ cùng các câu | 84,68 / 84,04 | 80,74 / 78,37 |
| Chỉ dùng ba tên riêng lẻ | 84,64 / 84,00 | 80,69 / 78,41 |

Có 4.181 và 4.010 chu trình trên hai split. Sai số khi triệt tiêu một hiệu thực sự sinh từ điểm đơn dưới 0,000000005. Một vector truy vấn 512 chiều tái hiện phép so sánh tường minh qua các tam giác. Đối chứng danh sách từ cũng cho tín hiệu lặp lại mạnh trên ảnh. Với cùng cấu hình, cú pháp chủ thể và nền chỉ nhỉnh hơn danh sách khoảng 0,02 và 0,01 điểm mAP. Phương pháp chưa vượt thành phần bắc cầu trên cả hai split. Điều được kiểm chứng là tồn tại một tín hiệu ngoài điểm từng từ, còn giá trị truy hồi riêng của tín hiệu đó chưa rõ.

Dùng 900 và 756 sketch mẫu với toàn gallery ở mỗi split. Đây là thử nghiệm thăm dò với bộ chấm vector hóa. Nhánh dừng ở pilot, chưa có số chấm đầy đủ trên TU-Berlin. [Cơ chế và bài báo](../../CYCLIC_DESCRIPTION_VERIFICATION.md) · [Tất cả cấu hình](pilot.json) · [Kết quả từng truy vấn](pilot-per-query.zip)
