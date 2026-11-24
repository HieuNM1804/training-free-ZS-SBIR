# Một từ của sketch cho biết phần nào của ảnh có thể giải thích một từ khác?

Mỗi ảnh ứng viên sinh 16 phân bố từ CLIP bằng cách giữ những tập token khác nhau và mã hóa lại. Một từ rút từ sketch cung cấp bằng chứng cho tập phần ảnh đã được thể hiện. Dùng hậu nghiệm đó dự báo một **từ khác**, chấm theo phân bố của một bản vẽ độc lập cùng nguồn. Đối chứng quan trọng giữ nguyên phân bố hậu nghiệm về *số vùng* được giữ nhưng chọn lại ngẫu nhiên các vùng trong từng nhóm kích thước. Mỗi dự báo chỉ quan sát một từ của sketch; 128 lượt rút ước lượng kỳ vọng. Mỗi lần chấm dùng đủ phân bố trên 20.931 từ sau khi loại từ đã quan sát.

| Log score dự báo bản vẽ khác, mô hình dùng cùng phần ảnh trừ đối chứng (nat) | Ext-1 | Ext-2 |
|---|---:|---:|
| Ảnh đúng, so với đọc ảnh nguyên vẹn, nhóm cùng lớp | +0,7060 | +0,7377 |
| Ảnh đúng, so với chọn hai tập vùng độc lập | +0,1003 | +0,1051 |
| Ảnh đúng, so với giữ số vùng và thay vùng | **+0,02950** | **+0,02938** |
| Biên đúng trừ sai so với giữ số vùng, đối thủ cùng lớp | +0,00110 | −0,00341 |
| Biên đúng trừ sai so với giữ số vùng, đối thủ khác lớp | **−0,04600** | **−0,04452** |

Khoảng bootstrap 95% theo cặp cho lợi ích ở đúng ảnh so với giữ số vùng là [0,0206, 0,0387] / [0,0183, 0,0403]. Như vậy từ đầu tiên cho biết một phần về *danh tính vùng*, ngoài thông tin về kích thước. Tuy nhiên, với đối thủ khác lớp, khoảng của chênh lệch biên là [−0,0585, −0,0333] / [−0,0577, −0,0319]. Dùng một phần của ảnh để giải thích từ của sketch còn giúp ảnh sai điều chỉnh dự báo nhiều hơn ảnh đúng. Trong nhóm cùng lớp, xác suất chọn đúng ảnh nguồn là 54,33% / 53,57%, với đối chứng lấy hai tập vùng độc lập là 55,67% / 53,97%.

Trên ảnh đúng ở nhóm cùng lớp, một từ đem lại 0,3001 / 0,2821 nat thông tin theo KL về tập vùng. Phần thay đổi chỉ do kích thước là 0,1328 / 0,1301; phần còn lại là 0,1673 / 0,1521. Đây là thông tin trong mô hình CLIP với tập vùng cố định. Người vẽ không cung cấp nhãn cho các vùng thật sự bị lược. [Thiết kế và prior art](protocol.md) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [Dữ liệu theo cặp](paired-T4.zip)
