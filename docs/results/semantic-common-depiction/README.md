# Hai phần sketch có thật sự cùng chọn một cách lược thông tin trong ảnh?

Chia sketch thành hai vùng tách biệt, chừa một hàng hoặc cột patch ở giữa. Hai vùng dùng hai nửa từ vựng khác nhau. Với mỗi vùng, bằng chứng là *từ CLIP có thể đọc ở sketch trùng từ ảnh ứng viên có thể sinh ra*. Danh tính của từ ấy chưa biết. Ứng viên phải giải thích cả hai lần trùng bằng một tập vùng tiềm ẩn duy nhất. Đối chứng cho phép mỗi lần trùng sinh ra từ một tập vùng độc lập. Cả hai giả thuyết có likelihood giống hệt nhau với mọi quan sát đơn lẻ. Vì vậy thay đổi thứ hạng chỉ có thể đến từ việc hai quan sát chia sẻ một cách diễn giải.

| Thử nghiệm trên các cặp gallery khác ảnh mẫu, một ảnh đúng lớp | Ext-1 | Ext-2 |
|---|---:|---:|
| Cặp đã chọn / sketch có đủ hai hướng để tính độ chính xác cân bằng | 850 / 195 | 998 / 362 |
| Độ chính xác cân bằng, chung tập vùng | 58,21% | 47,51% |
| Độ chính xác cân bằng, tập vùng độc lập | 58,21% | 47,51% |
| Độ chính xác cân bằng, chỉ chung kích thước tập vùng | 58,21% | 47,51% |
| Thay đổi biên log likelihood ảnh đúng trừ ảnh sai so với độc lập | −0,02308 | −0,00046 |
| Số quyết định độc lập sai được sửa / đúng bị đảo sai, trên tất cả các cặp | 5 / 3 | 9 / 6 |

Các phương án có thể đổi quyết định cá biệt trong khi cho cùng độ chính xác trên nhóm sketch đủ hai hướng. Trên dữ liệu này, yêu cầu một tập vùng chung không có lợi ích cân bằng so với các đối chứng. Trên tập cặp mẫu, việc hai phần của sketch có cùng cách phản hồi với một phần ảnh chưa giúp chọn đúng lớp. [Thiết kế và đẳng thức kiểm tra được](protocol.md) · [Tính giới hạn khi cho phép mọi quan hệ giữa hai tập vùng](../semantic-dependence-limits/protocol.md) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [Kết quả theo cặp](paired-T4.zip)
