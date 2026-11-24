# Ảnh có giới hạn được các cách diễn giải sketch hay không?

Mỗi ảnh đề xuất sáu hướng thay đổi nghĩa khi hai vùng cùng xuất hiện. Sketch bộc lộ thay đổi log odds trên một nửa từ vựng. Giải bình phương tối thiểu theo từng cặp sketch–ảnh để tìm tổ hợp trong các hướng mà ảnh đề xuất. Chấm dự báo trên nửa từ còn lại, rồi đổi vai hai nửa. Dùng đối chứng có cùng số hướng từ các hiệu ứng cộng tính của chính ảnh. Tất cả các phương án cùng nhìn thấy các log odds dùng cho suy luận.

| Nhóm cùng lớp, sketch gốc, ngữ cảnh trung tâm | Ext-1 | Ext-2 |
|---|---:|---:|
| Cải thiện log score khi bổ sung sáu hướng tương tác vào sáu hướng truy vấn (nat) | +0,01043 | +0,00889 |
| Cùng phép so sánh, lợi ích trên biên phân biệt nguồn (nat) | +0,00088 | +0,00093 |
| Log score trừ đối chứng cùng số hướng cộng tính của ảnh (nat) | −0,00094 | −0,00107 |
| Biên phân biệt nguồn trừ đối chứng cùng số hướng (nat) | +0,00033 | +0,00094 |
| Chọn đúng nguồn (mỗi sketch độc lập) | 49,00% | 54,37% |

Trong nhóm khác lớp, đưa thông tin ảnh vào cải thiện biên so với chỉ dùng truy vấn trên cả hai split. Nhưng khi đối chiếu với cùng số hướng cộng tính từ ảnh, thay đổi biên là −0,00187 / −0,00133 nat. Lợi ích chưa chứng minh được vai trò riêng của tương tác giữa các vùng.

Một điểm hạn chế quan trọng là hơn 10.000 logit trong mỗi nửa quan sát đều đến từ cùng một embedding 512 chiều. Chỉ giữ riêng tên từ chưa đảm bảo giá trị cần dự báo là thông tin thực sự còn thiếu. Thí nghiệm kế tiếp kiểm tra hạng ma trận từ vựng và việc có thể giải ngược embedding mà không dùng ảnh. [Thiết kế và đối chứng](protocol.md) · [Kiểm tra chỉ bộc lộ một bit](../regional-one-bit-inference/protocol.md) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [Điểm theo từng cặp](paired-T4.zip)
