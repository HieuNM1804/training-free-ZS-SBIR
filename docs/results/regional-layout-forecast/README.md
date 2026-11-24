# Kiểm tra bố cục chưa biết trên ảnh nguồn khác

Đã loại hợp mọi ảnh nguồn dùng ở thí nghiệm trước trong cả hai split. Cùng dùng các crop mã hóa riêng ở hai miền. Mỗi giả thuyết bố cục dự báo một vùng sketch trước khi CLIP đọc vùng đó. Các số dưới là tỷ lệ ghép đúng hai nhóm bản vẽ với hai ảnh nguồn cùng lớp, tính trung bình đều theo cặp (200 cặp / 2.317 sketch ở Ext-1, 168 cặp / 2.055 sketch ở Ext-2).

| Cách dự báo vùng được giữ lại | Ext-1 | Ext-2 |
|---|---:|---:|
| Trộn đều các vị trí, chưa thấy vùng khác | 57,35% | 55,80% |
| Quan sát một vùng để suy ra bố cục | 58,14% | 56,19% |
| Quan sát ba vùng, suy ra một trong tám phép đối xứng của hình vuông | 58,10% | 55,78% |
| Quan sát ba vùng, cho phép cả 24 hoán vị | 58,03% | 55,65% |
| Giữ tọa độ gốc khi dự báo | 73,64% | 68,65% |

Tín hiệu vị trí ở các crop mã hóa riêng lặp lại trên ảnh mới, nhưng mô hình dùng ba vùng chưa suy ra bố cục tốt hơn đối chứng một vùng. Nó giảm log score của vùng được giữ lại so với trộn đều −0,0279 / −0,0537 nat. Với đối thủ khác lớp, nó tăng biên dự báo ảnh đúng trừ ảnh đối thủ +0,0722 / +0,0746 nat; mức tăng đến từ giảm điểm đối thủ nhiều hơn, trong khi chất lượng dự báo ảnh đúng thực tế đi xuống. So với mọi hoán vị, lợi ích riêng của tám phép đối xứng không ổn định. Cách đọc cùng tọa độ dự báo tuyệt đối kém hơn trên ảnh thật nhưng phân biệt được đối thủ tốt hơn. Những con số này không chứng minh vị trí tương ứng với cùng bộ phận vật thể hoặc chuyển được sang bộ dữ liệu khác.

[Mô hình xác suất và thiết kế khóa](protocol.md) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [Dữ liệu ghép cặp](paired-T4.zip) · [Thiết kế kiểm tra trên các ca ngữ nghĩa toàn ảnh ghép sai](complementarity-protocol.md)

### Bằng chứng bổ sung khi ngữ nghĩa toàn ảnh ghép sai

Đối với các phép ghép hai nguồn cùng lớp mà điểm từ toàn ảnh ghép sai, đọc crop cùng tọa độ ghép đúng **74,17% / 68,20%**. Giữ nguyên mọi descriptor và lấy trung bình điểm qua tám cách đặt hệ tọa độ cho kết quả **58,49% / 54,80%**. Chênh lệch ghép cặp là **+15,68** [11,69, 19,63] và **+13,40** [8,86, 18,04] điểm phần trăm. Có 195 và 163 cặp ảnh đóng góp vào hai ước lượng. Mô hình tự suy ra hướng đạt **58,10% / 51,66%** trong cùng nhóm. Khoảng tin cậy 95% bootstrap theo cặp trong từng split. Hai split có nguồn chung nên các ước lượng giữa split có thể phụ thuộc.

Độ chính xác chọn ảnh đúng của **một sketch riêng** trên mọi cặp cùng lớp chỉ là **57,32% / 52,90%** ở tọa độ gốc và **53,26% / 53,43%** với ngữ nghĩa toàn ảnh. Phép ghép sau khi chấm khử thiên lệch cố định của một ảnh trong cặp. Phân tích nhóm lỗi này được thiết kế sau khi đã xem kết quả trên các ảnh mới và mang tính chẩn đoán. Số liệu cho thấy có thông tin vị trí bổ sung trong cách đặt bản vẽ Sketchy cạnh chính ảnh được dùng làm mẫu. Xác định được vị trí tương ứng khi hình được vẽ từ góc khác vẫn là bài toán mở. [Số liệu và mọi đối chứng](conditional-information.json) · [Biểu đồ khoảng tin cậy](conditional-information.pdf)
