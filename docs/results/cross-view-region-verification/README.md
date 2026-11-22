# Đọc vùng ảnh cho từng ứng viên và kiểm tra qua hai góc nhìn

Với mỗi ảnh ứng viên, CLIP đo mức từng vùng của sketch ủng hộ các từ đại diện cho ảnh đó. Ta có thể chọn vùng bằng sketch gốc rồi đọc vùng tương ứng sau khi lật, hoặc dùng cùng một góc nhìn cho cả hai bước. Cả hai cách đều chạy CLIP trên cùng 13 vùng ở mỗi góc nhìn. Trọng số của tín hiệu vùng đã khóa là 0,35.

| mAP / precision (%) | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin 30 lớp |
|---|---:|---:|---:|
| Từ công khai và diffusion gallery RGB | 81,86 / 81,33 | 74,19 / 72,68 | 79,16 / 80,24 |
| Trung bình đều các vùng qua hai góc nhìn | 84,09 / 83,70 | 76,47 / 74,27 | 79,83 / 80,74 |
| Chọn vùng ở một góc nhìn, đọc ở góc còn lại | 84,25 / 83,72 | 76,87 / 74,69 | 79,91 / 80,67 |
| Chọn và đọc trong cùng góc nhìn | 84,37 / 83,85 | 77,01 / 74,84 | 80,07 / 80,92 |

So với chọn và đọc trong cùng góc nhìn, bước kiểm tra qua ảnh lật làm giảm 0,113, 0,140 và 0,165 điểm mAP. Khoảng tin cậy 95% bootstrap theo lớp tương ứng là [−0,187; −0,047], [−0,212; −0,072] và [−0,294; −0,055] điểm. Trên TU-Berlin, kiểm tra qua ảnh lật tăng 0,079 điểm so với trung bình đều, với khoảng [−0,132; +0,285]. Bằng chứng ủng hộ chọn vùng theo ứng viên. Phép kiểm tra qua ảnh lật chưa cho thấy lợi ích riêng.

Chọn cấu hình trên hai split Sketchy unseen, dùng trung bình mAP rồi đến precision khi hòa. TU-Berlin dùng cấu hình đã khóa. Cả hai benchmark đều đo chiều sketch → photo. OpenAI CLIP ViT-B/32 được đóng băng. Hàm KD-SBIR gốc chấm trên đủ 14.179, 12.694 và 2.130 sketch cho mỗi cấu hình cuối. Ext-1 và TU dùng mAP@all và P@100. Ext-2 dùng mAP@200 và P@200. Các nhánh dùng 20.931 tên công khai Open Images. Hai split Sketchy đã được dùng để chọn tham số. [Giao thức](../../CROSS_VIEW_REGION_PROTOCOL.md) · [Khóa](freeze.json) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU](tuberlin.json) · [Từng truy vấn](per-query.zip)
