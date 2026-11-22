# Quan hệ giữa các khái niệm còn giữ được bao nhiêu từ ảnh sang sketch?

Giữ nguyên những từ CLIP đề xuất cho sketch. Với từng ảnh, quan sát hai hướng của 13 vùng để đo xem hai từ có cùng tăng hoặc giảm khi chuyển vùng hay không. Từ mỗi quan hệ, trừ phần dự kiến chỉ do hai embedding văn bản gần nhau, rồi trừ phần có thể quy về điểm riêng của từng từ. Phép đo còn lại có thể phân biệt hai ảnh có cùng phân bố điểm của từng từ nhưng các từ xuất hiện cùng nhau ở những vùng khác nhau. Chúng tôi dùng Kendall concordance cho phép đo và kiểm tra đối chứng hoán vị độc lập các vùng của mỗi từ. Hoán vị giữ nguyên phân bố của từng từ và cả đáp ứng của nó khi ảnh bị lật. [Công thức, giả định và tiền lệ](../../PAIRED_CONCEPT_RESPONSE_GEOMETRY.md)

## Khi nào tín hiệu này nhận ra vật thể?

Mỗi truy vấn có một ảnh gốc được xác nhận bằng ID trong Sketchy. Các phương pháp so sánh ảnh gốc với một ảnh khác cùng lớp ở gần vị trí xếp hạng của ảnh gốc. Tỷ lệ dưới đây là phần trăm chọn đúng trong phép thử hai lựa chọn, dùng 16 tên do sketch cung cấp.

| Hai quan sát cần đối chiếu | Ext-1 (900 sketch) | Ext-2 (756 sketch) |
|---|---:|---:|
| Ảnh gốc ở tư thế thường và chính ảnh đó khi lật | 89,89 | 89,42 |
| Các vùng nhỏ của ảnh gốc và các vùng lớn của cùng ảnh | 61,33 | 62,83 |
| Sketch và ảnh gốc, đọc cả hai tư thế ở cả hai phía | 51,56 | 54,50 |

Đối chứng hoán vị ở phép ảnh–ảnh lật chỉ đạt 48,89% và 49,34%. Như vậy phép đo thực sự tìm thấy sự tương ứng giữa các từ trong cùng ảnh. Ngay cả khi còn cùng ảnh, thay kích thước vùng quan sát cũng làm giảm độ nhận diện. [Số liệu từng điều kiện và từng query](instance-transfer-audit.json) · [Bảng số từng sketch](instance-transfer-audit.zip)

Một phép thử riêng giữ nguyên cả ba vật thể và thay độc lập miền ảnh thật cùng từ mô tả miền. Ở phép này mỗi bên dùng một tư thế. Khi giữ cố định prompt nguồn “sketch” và prompt đích “photo”, việc nhận ra cùng một ảnh lật đạt 88,00% / 88,10%. Hai bản vẽ khác nhau của cùng vật thể chỉ đạt 55,11% / 53,44%. Dùng sketch tìm ảnh gốc đạt 49,33% / 51,98%. Thay cả hai prompt bằng “photo” cho 51,44% / 52,51%. Các phép đo chưa cho thấy đổi tên miền trong prompt đủ để phục hồi quan hệ bị mất khi vẽ lại. Các bản vẽ khác và ID ghép vật thể chỉ phục vụ phân tích sau xếp hạng. [Thí nghiệm chéo miền hình ảnh và prompt](actual-depiction-factorial.json) · [Đối tượng đã ghép](actual-depiction-factorial.zip)

## Kết quả truy hồi theo lớp

| mAP / precision (%) | Sketchy Ext-1 | Sketchy Ext-2 |
|---|---:|---:|
| Mốc mạnh cùng T4 | 84,64 / 83,94 | 80,64 / 78,38 |
| Quan hệ giữa từ sau khi trừ hình học và hiệu ứng từng tên | 84,63 / 83,95 | 80,63 / 78,37 |
| Phá quan hệ bằng hoán vị, chọn riêng trên hai split | 84,63 / 83,94 | 80,60 / 78,38 |

Cùng mốc đề cử 512 ảnh, mọi điều kiện được phép xếp lại chính 512 ảnh ấy, toàn bộ gallery vẫn được tính trong metric. Cấu hình chính chọn 8 tên, hệ số 0,06. Chênh mAP so với mốc là −0,002 và −0,018 điểm. Trên những cặp đầu tiên mà mốc xếp ảnh sai ngay trên ảnh đúng, cấu hình phân tích 16 tên chọn ảnh đúng 49,2% và 51,5%. Đây là kết quả thăm dò 36 sketch mỗi lớp trên toàn bộ gallery, được chấm bằng evaluator KD-SBIR gốc. Giá trị thực nghiệm của nhánh này nằm ở phép đo mất thông tin qua các quan sát. Lợi ích truy hồi và khả năng chuyển sang bộ khác vẫn chưa được xác lập. [Tất cả cấu hình và đối chứng](pilot.json) · [Từng sketch](pilot-per-query.zip) · [Các cặp đúng–sai](pair-diagnosis.json)
