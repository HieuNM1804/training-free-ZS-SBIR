# Khi hai ảnh sát hạng, phần nào của điểm vùng hữu ích?

Mã tái lập từng điểm AP và precision của Candidate-Conditioned Regional Matching gốc trên cùng 768 truy vấn mỗi split. Mỗi cấu hình dùng cùng từ điển vùng và ảnh ứng viên. Ta thay quan sát sketch hoặc thay ảnh quyết định trọng số các vùng. Trong đối chứng sát hạng, ảnh chọn vùng là láng giềng của ứng viên theo điểm toàn sketch trong từ điển vùng. Ảnh được chấm luôn là ứng viên ban đầu.

| Quan sát và cách chọn vùng | Ext-1 mAP@all / P@100 | Ext-2 mAP@200 / P@200 | Độ chính xác cân bằng ở các cặp sát hạng, Ext-1 / Ext-2 |
|---|---:|---:|---:|
| F · Toàn sketch, cùng từ điển vùng | **81,680 / 80,367** | **73,906 / 72,369** | 60,67% / 65,10% |
| U · Các crop, trọng số vùng cố định | 78,165 / 77,182 | 68,763 / 67,033 | 58,56% / 61,06% |
| R · Mỗi ứng viên chọn vùng cho chính nó | 78,302 / 76,826 | 68,414 / 67,156 | 62,38% / 64,17% |
| M · Ảnh đứng sát hạng chọn vùng cho ứng viên | 78,419 / 77,147 | 68,758 / 67,466 | 59,44% / 63,44% |

Với mỗi tư thế sketch, gọi ℓᵢ là độ khớp của ảnh ứng viên ở vùng i, πᵢ là xác suất vùng ban đầu và rᵢ là ℓᵢ chia cho độ khớp trung bình của gallery tại vùng đó. Trước khi lấy trung bình hai tư thế và lấy log, thay đổi từ U sang R bằng Covπ(r,ℓ) / Eπ[r]. Một ứng viên vì thế có thể tự tăng điểm khi vùng mà nó nổi bật hơn các ảnh khác cũng là vùng nó có độ khớp cao. Ưu tiên ấy có thể xuất hiện cho cả ứng viên đúng lẫn ứng viên sai.

Điểm log có phân rã đúng R−F = (U−F) + (R−U). Xét riêng dấu của thay đổi biên giữa một ảnh đúng và một ảnh sai, R−U phân biệt đúng 68,56% / 63,36% trên chỉ số cân bằng; U−F đạt 50,63% / 44,41%. Khi đối chứng bằng đúng ảnh đứng sát hạng, phần R−M còn 51,48% / 49,01%. Chỉ số cân bằng sử dụng 59 / 157 sketch có đủ hai hướng sai và đúng của quyết định toàn ảnh. Trên full gallery, mAP R−U là +0,136 / −0,349 điểm và R−M là −0,117 / −0,344 điểm. Điểm vùng chưa chứng minh được lợi ích truy hồi phụ thuộc vào danh tính của chính ứng viên. [Thiết kế và prior art](protocol.md) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [Kết quả theo truy vấn](paired-T4.zip)
