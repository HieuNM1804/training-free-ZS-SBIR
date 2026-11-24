# Bao nhiêu lỗi có thể sửa nếu chọn tốt nhất cách liên hệ hai phần sketch?

Giữ bằng chứng của từng phần và phân bố 16 cách lược vùng của mỗi ứng viên. Cho phép mọi phân bố đối xứng trên cặp cách lược vùng, miễn là mỗi quan sát riêng vẫn có đúng phân bố biên đã khóa. Với mỗi cặp sketch–ảnh, tìm log likelihood nhỏ nhất và lớn nhất trên tập này. Đây là hai bài toán vận chuyển hữu hạn với hàm mục tiêu tuyến tính. Cận đối ngẫu được hiệu chỉnh để thỏa ràng buộc trên toàn ma trận. Chênh tối đa giữa nghiệm nguyên thủy và chứng chỉ đối ngẫu sau chuẩn hóa nhỏ hơn 6×10⁻¹⁰. Hai phương án dùng chung và dùng độc lập đều nằm trong khoảng thu được trên mọi cặp.

| Cặp liền kề có một ảnh đúng lớp | Sketchy Ext-1 | Sketchy Ext-2 |
|---|---:|---:|
| Tổng số cặp | 850 | 998 |
| Cặp có thứ tự giữ nguyên dưới mọi quan hệ được phép | 814 | 947 |
| Lỗi của mô hình coi hai quan sát độc lập | 242 | 483 |
| Lỗi không thể đảo chiều bằng quan hệ giữa hai cách lược vùng | **226 / 242 (93,4%)** | **458 / 483 (94,8%)** |
| Lỗi mà ít nhất một quan hệ còn có thể sửa | 16 | 25 |
| Độ chính xác cân bằng của mô hình độc lập | 58,21% | 47,51% |
| Trần cân bằng nếu biết nhãn và chọn quan hệ có lợi nhất cho từng cặp | **60,00%** | **49,72%** |

Độ chính xác cân bằng chỉ dùng 195 / 362 sketch có cặp thuộc cả hai hướng của thứ hạng mốc. Ngay cả khi cho phép chọn quan hệ riêng cho từng sketch–ảnh sau khi biết đáp án, dư địa chỉ còn +1,79 / +2,21 điểm phần trăm trên tập đó. Trên các cặp có thứ tự được xác định, tỷ lệ đúng là 72,24% / 51,64%; nếu chỉ lấy các quyết định độc lập tự tin nhất ở cùng độ phủ, tỷ lệ là 71,99% / 51,53%. Vì vậy một thứ tự bất biến với quan hệ giữa các vùng vẫn có thể là thứ tự sai.

Giới hạn này áp dụng cho đúng các quan sát CLIP, dạng likelihood, phân bố biên và các cặp đã lấy mẫu. Nó không đặt trần cho mAP, cho mô hình đổi nội dung đọc từ ảnh, hay cho ZS-SBIR nói chung. Đóng góp thực nghiệm là phân biệt được lỗi do chưa biết cách kết hợp dữ kiện với lỗi đã có trong chính các dữ kiện đưa vào. [Công thức và prior art](protocol.md) · [Đổi có kiểm soát các dữ kiện biên](../semantic-marginal-error-audit/protocol.md) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [Khoảng và quyết định từng cặp](paired-T4.zip)
