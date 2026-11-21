# Một từ công khai có truyền được hai ngữ cảnh thị giác từ photo sang sketch?

Với mỗi từ trong chỉ mục, một nửa gallery đề cử hai từ ngữ cảnh ở hai đầu của một trục biến thiên thị giác. CLIP mã hóa chúng cùng từ gốc. Sketch quyết định cách phân bổ bằng chứng giữa các ảnh của nửa gallery còn lại. Tổng bằng chứng của mỗi từ được bảo toàn.

| mAP / precision (%) | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin 30 lớp |
|---|---:|---:|---:|
| Phân bổ theo hai cụm từ | 82,67 / 82,13 | 75,77 / 73,75 | 77,97 / 78,73 |
| Giữ nguyên cách đọc từ gốc | 82,86 / 82,18 | 75,95 / 73,85 | 78,32 / 78,86 |
| Đảo cách phân bổ | 82,76 / 82,14 | 75,51 / 73,83 | 78,16 / 78,50 |
| Cộng hai embedding từ thay cho mã hóa cụm từ | 82,74 / 82,22 | 75,87 / 73,82 | 78,15 / 78,83 |
| Khớp từ + diffusion RGB | 81,86 / 81,33 | 74,19 / 72,68 | 79,16 / 80,24 |

Trên ảnh giữ lại ở hai split Sketchy, hướng tương phản của cụm từ đồng biến với trục thị giác ở 98,4–100% số từ theo fold. Trên sketch, phương pháp chính thấp hơn đối chứng dùng từ gốc 0,19, 0,18 và 0,35 điểm mAP. Phân biệt được hai nhóm photo chưa đủ để suy ra một sketch sẽ chọn đúng nhóm. Hai kết quả dùng cùng từ điển, prompt miền, chỉ mục gallery và công thức trộn điểm.

Chọn cấu hình trên hai split Sketchy unseen, dùng trung bình mAP rồi đến precision khi hòa. TU-Berlin dùng cấu hình đã khóa. Cả hai benchmark đều đo chiều sketch → photo. OpenAI CLIP ViT-B/32 được đóng băng. Hàm KD-SBIR gốc chấm trên đủ 14.179, 12.694 và 2.130 sketch cho mỗi cấu hình cuối. Ext-1 và TU dùng mAP@all và P@100. Ext-2 dùng mAP@200 và P@200. [Giao thức](../../GALLERY_WORD_SENSES.md) · [Khóa](freeze.json) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU](tuberlin.json) · [Từng truy vấn](per-query.zip)
