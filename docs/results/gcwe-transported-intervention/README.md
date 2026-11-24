# Hình học nhận ra ảnh nguồn có dự báo được thay đổi nghĩa khi hoán đổi hai phần?

Mỗi lần hoán đổi giữ nguyên toàn bộ 49 patch. Ảnh ứng viên phải dự báo bằng chứng từ vựng sẽ thay đổi thế nào khi đổi chỗ hai phần của sketch. Từ cặp ảnh nguyên vẹn, tính phân bố trên tám ánh xạ bảo toàn hình vuông rồi chuyển phép đổi chỗ của sketch sang hệ tọa độ của ảnh. Nếu ánh xạ là g và phép đổi chỗ là τ, tác động tương ứng trên ảnh là gτg⁻¹. Có ba phép so sánh, mỗi phép gồm hai hoán đổi dùng bốn vùng tách biệt. Khi chấm, rút hai từ từ các phiên bản sketch đã đổi chỗ và yêu cầu ứng viên gán mỗi từ về đúng hoán đổi. Toàn bộ từ vựng công khai gồm 20.931 từ. Không huấn luyện, dùng seed 42.

| Sketch nguyên gốc, đối thủ cùng lớp | Ext-1 | Ext-2 |
|---|---:|---:|
| Số cặp nguồn, mỗi nguồn cung cấp một sketch | 150 | 126 |
| Chọn đúng ảnh nguồn theo xác suất của **ánh xạ khả dĩ nhất** | **59,00%** | **56,35%** |
| Chọn đúng nguồn theo dự báo từ sau khi chuyển phép hoán đổi | 49,33% | 46,03% |
| Chọn đúng nguồn khi cố định tọa độ tương ứng | 57,67% | 61,11% |
| Gán đúng từ về phép hoán đổi, dùng phân bố ánh xạ của đúng ảnh | 51,54% | 51,11% |
| Giới hạn tham chiếu khi biết phân bố từ của chính sketch đã hoán đổi | 78,55% | 77,60% |

Phép so sánh theo từng giai đoạn là phân tích hậu nghiệm trên đúng các cặp đã chấm. Khoảng bootstrap 95% theo cặp ảnh của hàng độ tin cậy ánh xạ là [53,67%, 64,33%] / [51,19%, 61,90%]. Khi chuyển sang dự báo bằng từ, độ chính xác chọn nguồn giảm 9,67 / 10,32 điểm, với khoảng [−18,33, −1,00] / [−19,06, −1,59] điểm. Vậy CLIP vẫn mang thông tin về sự tương ứng giữa hai ảnh trước khi dùng nó để suy ra nghĩa của thay đổi bố cục. Dư địa ở hàng cuối cho thấy phép thao tác ảnh thực sự làm thay đổi phản hồi từ của sketch.

Khi chỉ lật ngang sketch, suy ánh xạ cải thiện biên đúng trừ sai so với ghép cứng theo tọa độ cũ. Kết quả chưa vượt mô hình cho phép cả 24 hoán vị tự do ở cả hai split theo tiêu chí đã khóa. Những thao tác này là can thiệp định lượng vào CLIP, và đánh giá chọn ảnh nguồn khác với chấm AP theo lớp. [Thiết kế và prior art](protocol.md) · [Tách bằng chứng hình học và ngữ nghĩa](stage-localization.json) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [Dữ liệu theo cặp](paired-T4.zip)
