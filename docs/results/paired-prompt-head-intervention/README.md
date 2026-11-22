# Ổn định điểm giữa prompt “sketch” và “photo” có giúp truy hồi?

Với từng sketch, CLIP đề xuất các tên trong từ điển công khai. Phương pháp thay đổi trọng số đóng góp của 48 attention head trong biểu diễn cuối để giảm phần chênh lệch giữa hai prompt thay đổi theo từng tên. Một nửa số tên dùng để ước lượng mức điều chỉnh. Nửa kia quyết định nên giữ bao nhiêu phần điều chỉnh đó. Riêng 128 tên có hạng 193–320 chỉ dùng để kiểm tra xem hiệu ứng có lan sang tên mới. Điểm theo vùng và đồ thị ảnh của phương pháp nền giữ cùng cấu hình. [Cơ chế, công thức và nguồn](../../PAIRED_PROMPT_HEAD_INTERVENTION.md)

| mAP / precision (%) | Sketchy Ext-1 | Sketchy Ext-2 |
|---|---:|---:|
| Điểm nền theo vùng | 84,64 / 83,96 | 80,63 / 78,38 |
| Giảm chênh lệch prompt bằng các head của sketch | 84,54 / 83,80 | 80,74 / 78,42 |
| Tăng chênh lệch, cùng hệ số với hàng trên | 84,69 / 83,91 | 80,52 / 78,23 |
| Tăng chênh lệch, dò hệ số riêng | 85,00 / 84,52 | 80,62 / 78,11 |
| Chỉ nội suy giữa hai prompt, dò hệ số riêng | 84,69 / 84,00 | 80,53 / 78,28 |
| Không gian từ vựng có cùng hình học với các head, dò hệ số riêng | 84,69 / 84,08 | 80,61 / 78,24 |

Cấu hình giảm chênh lệch dùng 64 tên, ridge 0,1 và trọng số 0,3 cho phần sai lệch miền. Nó giảm phương sai chênh lệch giữa hai prompt trên các từ giữ lại 5,74% ở Ext-1 và 7,27% ở Ext-2. Tương quan Pearson giữa mức giảm này và thay đổi AP thực tế theo sketch chỉ đạt 0,049 và −0,018. Hệ số tương quan của điểm nội dung trước và sau can thiệp trên các từ đó đạt 0,824 và 0,792. Trên 127 truy vấn Ext-1 có kết quả đầu sai và có tên lớp công khai khớp chính tả, tên lớp được đẩy lên ở 38 truy vấn và tụt hạng ở 83. Trên Ext-2, các con số tương ứng là 65 và 60 trên 133 truy vấn. Việc làm cho hai prompt đồng nhất hơn đôi khi xóa tín hiệu giúp phân biệt lớp.

Mức thay đổi mAP so với nền là −0,096 điểm trên Ext-1 với khoảng tin cậy bootstrap 95% theo lớp [−0,220, +0,002]. Ở Ext-2, mức thay đổi là +0,111 [−0,119, +0,367]. Trung bình hai split tăng 0,008 điểm. Mẫu này chưa đủ cơ sở chọn giảm chênh lệch prompt làm phương pháp chính hoặc đưa nó sang TU-Berlin.

Mỗi lớp đóng góp 36 sketch. Ext-1 gồm 900 sketch và 17.100 ảnh, Ext-2 gồm 756 sketch và 12.552 ảnh. Cấu hình cuối và các đối chứng trong bảng đều được chấm bằng hàm KD-SBIR gốc cho từng truy vấn trên Colab L4. Ext-1 dùng mAP@all và P@100, Ext-2 dùng mAP@200 và P@200. Đây là mẫu cân bằng lớp và việc chọn hệ số trên unseen là thăm dò. Điểm đầy đủ chịu trọng số khác do số sketch mỗi lớp khác nhau. [Tất cả ablation](pilot.json) · [Ext-1](pilot-sketchy_1.json) · [Ext-2](pilot-sketchy_2.json) · [Điểm và chẩn đoán từng truy vấn](pilot-per-query.zip)
