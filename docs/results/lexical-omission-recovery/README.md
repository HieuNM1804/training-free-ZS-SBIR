# Khôi phục phần thông tin gallery mà 64 từ chưa giải thích

Mỗi từ công khai dự báo một phân bố trên các nhóm ảnh không nhãn. Một bài toán tuyến tính tính phần lớn nhất có thể giải thích bằng 64 từ trong chỉ mục, đồng thời trên hai nửa gallery. Sketch đề xuất từ bằng CLIP đóng băng. Phần xác suất còn thiếu được phân bổ cho những nhóm ảnh tương ứng. Thứ tự bên trong mỗi nhóm giữ nguyên. Cấu hình đã khóa đọc 4 từ dựa trên các vùng sketch, hệ số 0,3 cho xác suất từ và 0,04 cho mức cập nhật.

| mAP / precision (%) | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin 30 lớp |
|---|---:|---:|---:|
| Mốc vùng và từ | 84,37 / 83,85 | 77,01 / 74,84 | 80,07 / 80,92 |
| Chỉ bổ sung phần chưa được hỗn hợp 64 từ giải thích | 84,26 / 83,78 | 77,02 / 74,78 | 80,43 / 80,86 |
| Bổ sung toàn bộ phân bố, cùng từ và hệ số | 84,21 / 83,72 | 76,88 / 74,66 | 80,48 / 80,93 |
| Bổ sung toàn bộ phân bố với cùng khối lượng thay đổi | 84,29 / 83,78 | 76,93 / 74,76 | 80,44 / 81,03 |
| Bổ sung trực tiếp, chọn hệ số và cách đọc độc lập | 84,33 / 83,81 | 76,93 / 74,77 | 80,33 / 81,09 |
| Hoán vị phần thiếu của các từ, bảo toàn lượng thiếu mỗi từ | 84,34 / 83,82 | 76,97 / 74,78 | 80,05 / 80,81 |

So với mốc vùng, chênh lệch mAP là −0,108, +0,009 và +0,352 điểm, với khoảng tin cậy 95% theo lớp trên TU [−0,202; +1,204]. Trên TU, đối chứng dùng cả phân bố với cùng cấu hình đạt 80,48 mAP và đơn giản hơn bước loại phần trùng. Trong các lỗi hạng 1 ban đầu, biến thể chính tăng 0,80 điểm AP@200 trên Ext-2, 1,14 điểm AP trên TU và giảm 0,31 điểm trên Ext-1. Nghiệm cho từng từ đều có chứng nhận đối ngẫu. Việc phân tách khối lượng là xác định được đối với chỉ mục đã cho, còn lợi thế đối với xếp hạng chưa ổn định. [Chẩn đoán 20.931 từ và giới hạn chọn một từ](open-name-diagnosis.json)

Chọn cấu hình trên hai split Sketchy unseen, dùng trung bình mAP rồi đến precision khi hòa. TU-Berlin dùng cấu hình đã khóa. Cả hai benchmark đều đo chiều sketch → photo. OpenAI CLIP ViT-B/32 được đóng băng. Hàm KD-SBIR gốc chấm trên đủ 14.179, 12.694 và 2.130 sketch cho mỗi cấu hình cuối. Ext-1 và TU dùng mAP@all và P@100. Ext-2 dùng mAP@200 và P@200. Quyết định về từ chỉ dựa vào ảnh và danh mục công khai. Mọi số dùng nhãn để chọn từ tốt nhất trong phép chẩn đoán đều là oracle hậu nghiệm. [Giao thức](../../LEXICAL_OMISSION_RECOVERY.md) · [Khóa](freeze.json) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU](tuberlin.json) · [Từng truy vấn](per-query.zip)
