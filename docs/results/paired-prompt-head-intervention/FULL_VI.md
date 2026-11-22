# Khuếch đại sự khác biệt giữa prompt mô tả sketch và photo

Lượt thăm dò cho thấy ổn định điểm khi đổi prompt gần như không cải thiện mAP trung bình trên hai mẫu cân bằng. Lượt đầy đủ kiểm tra chiều ngược của cùng can thiệp. Với từng sketch, chọn các tên công khai mà CLIP cho là khả dĩ, dùng sự khác biệt giữa hai mô tả của mỗi tên để tăng trọng số những hướng phân biệt cách thể hiện vật thể. Can thiệp nằm trong không gian các đóng góp của 48 attention head, với ràng buộc về phần điểm nội dung chung giữa các prompt. Nhánh chỉ sửa đầu vào từ vựng toàn ảnh. [Cơ chế](../../PAIRED_PROMPT_HEAD_INTERVENTION.md) · [Lý do đổi giả thuyết sau pilot](../../PAIRED_PROMPT_FULL_FOLLOWUP.md) · [Pilot trước khi đổi dấu](README.md)

| mAP / precision (%) | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin 30 lớp |
|---|---:|---:|---:|
| Mốc từ, vùng và diffusion gallery | 84,37 / 83,85 | 77,01 / 74,84 | **80,07 / 80,92** |
| Khuếch đại bằng hướng của các head | 84,83 / 84,41 | 77,27 / 75,09 | 79,65 / 80,16 |
| Cùng hướng, bỏ bước điều chỉnh độ mạnh bằng nửa từ vựng còn lại | **84,86 / 84,44** | 77,25 / 75,09 | 79,58 / 80,14 |
| Chỉ ngoại suy giữa hai prompt, hệ số chọn riêng | 84,24 / 83,59 | **77,34 / 75,19** | 79,84 / 80,73 |
| Chuyển cùng độ mạnh sang không gian hướng chỉ từ văn bản | 84,08 / 83,48 | 76,47 / 74,25 | 79,37 / 79,90 |
| Hoán đổi danh từ giữa hai prompt trong bước ước lượng | 84,48 / 83,96 | 77,11 / 74,97 | 79,96 / 80,64 |
| Giảm chênh lệch prompt, hệ số chọn riêng | 84,35 / 83,83 | 76,99 / 74,80 | 80,06 / 80,94 |

Trên Ext-1, phương pháp chính tăng 0,466 điểm mAP, khoảng tin cậy 95% theo bootstrap lớp [+0,006, +0,985]. Trên Ext-2, mức tăng là +0,255 [−0,612, +1,135]. Trên TU-Berlin, chênh lệch là −0,425 [−0,996, +0,067]. Ở vị trí đầu, lần lượt có 199, 252 và 24 lỗi được sửa, cùng 91, 186 và 38 truy vấn mất ảnh đúng. TU-Berlin và đối chứng bỏ bước điều chỉnh độ mạnh giới hạn kết luận về khả năng chuyển giao và đóng góp của bước đó.

99 cấu hình được xét trên đủ 14.179 và 12.694 sketch unseen, chọn chung theo trung bình mAP rồi precision. Cấu hình chính dùng 192 từ, ridge 1 và hệ số sai lệch 12. Cấu hình cùng các đối chứng được khóa trước lần chấm 2.130 sketch của 30 lớp TU-Berlin. Tất cả 9 thứ hạng cuối được chấm bằng hàm KD-SBIR gốc cho từng sketch với sai lệch số tối đa so với phép tính vector hóa dưới 0,0000003. Colab NVIDIA L4, OpenAI ViT-B/32 đóng băng, xử lý từng sketch độc lập. Ext-1 và TU-Berlin dùng mAP@all và P@100, Ext-2 dùng mAP@200 và P@200. Hai bộ dữ liệu đều đo sketch → photo. [Khóa chung](freeze.json) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU-Berlin](tuberlin.json) · [Từng truy vấn](per-query.zip)
