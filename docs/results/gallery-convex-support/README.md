# Suy luận từ gallery theo từng khái niệm

Ứng viên nhận bằng chứng cho một từ khi mức điểm đó được giữ ở mọi hướng trong mặt phẳng tiếp tuyến cục bộ của gallery. Sau khi kiểm tra các từ riêng, điểm truy hồi là kỳ vọng theo posterior 601 từ của sketch. Một phép dựng giữ nguyên mọi điểm Dot cho hai gallery cho điểm mới **0,45** khi có một cách đọc chung quanh ảnh và **0,01** khi hai phía chỉ hỗ trợ hai cách đọc khác nhau. Thí nghiệm thực dùng đúng OpenAI CLIP ViT-B/32 đóng băng, hai prompt baseline, gallery không nhãn và mỗi truy vấn độc lập. [Công thức, giả thuyết, prior art và giao thức](protocol.md)

## Kết quả full unseen

| Cấu hình | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin 30 lớp |
|---|---:|---:|---:|
| Dot đủ 601 từ | 75,35 / 77,10 | 72,66 / 71,08 | 73,46 / 73,98 |
| Chứng nhận từng từ trước khi tính Dot | **76,66 / 77,53** | **72,86 / 71,46** | **75,03 / 74,46** |
| Tráo hướng trong từng nhóm hạng láng giềng | 77,00 / 77,36 | 71,58 / 70,53 | 75,34 / 74,17 |
| Chứng nhận sau khi tính Dot | 76,76 / 77,34 | 73,03 / 71,41 | 75,13 / 74,32 |
| RGB diffusion trên cùng graph 16 láng giềng | 78,44 / 78,67 | 74,63 / 72,66 | 76,75 / 75,33 |

Các cặp số là mAP / precision (%). Ext-1 và TU-Berlin dùng mAP@all / P@100. Ext-2 dùng mAP@200 / P@200. Colab T4, seed 42, commit chạy 1e8a921. Đủ 14.179 / 12.694 / 2.130 truy vấn trên 17.100 / 12.552 / 27.262 ảnh. Hàm KD-SBIR gốc chấm từng truy vấn và khớp với phép chấm từ thứ hạng trên GPU. Dot khớp chính xác số đã lưu trên cả ba split. [Cấu hình khóa](freeze.json) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU-Berlin](tuberlin.json) · [Kết quả từng truy vấn](paired-T4.zip)

## Khi nào phép nội suy có thể hoạt động

PCA qua gốc của lượt đầu thường đưa toàn bộ láng giềng về cùng một nửa mặt phẳng. Số ảnh có đủ hướng khi cho mọi láng giềng trọng số 1 tăng từ **121 lên 17.066** trên Ext-1, **100 lên 12.529** trên Ext-2 và **4 lên 27.238** trên TU-Berlin sau khi ước lượng trục PCA từ các sai khác đã trừ trung bình láng giềng. Gốc dùng để kiểm tra bao lồi vẫn là ảnh ứng viên. Lượt đầu thay đổi mAP so với Dot lần lượt +0,00048 / +0,00103 / −0,00019 điểm. Lượt sửa dựa trên kiểm tra hình học ở hai gallery Sketchy và giữ cùng cấu hình cho TU-Berlin. [Lượt đầu và bản khóa tương ứng](initial/freeze.json) · [Kết quả từng truy vấn của lượt đầu](initial/paired-T4.zip)

Với tọa độ đã sửa, mức tăng mAP so với Dot là **+1,32 / +0,20 / +1,57** điểm. Tính trung bình theo lớp, khoảng bootstrap 95% lần lượt từ 0,81 đến 2,06, từ −0,61 đến 1,30 và từ 0,87 đến 2,42 điểm. Có 23/25, 13/21 và 22/30 lớp tăng AP. Số top-1 được sửa trên ba bộ là 340 / 351 / 23, cùng 111 / 204 / 13 top-1 đúng bị đổi thành sai. Đối chứng áp dụng bao lồi sau Dot giữ nguyên top-1 vì điểm hỗ trợ bị chặn bởi Dot cao nhất và quy tắc phá hòa ưu tiên Dot gốc.

Trong các ứng viên dưới ngưỡng điểm của hạng K ở Dot, tỷ lệ đúng lớp ở nhóm chỉ hướng thật đưa qua ngưỡng so với nhóm chỉ hướng tráo đưa qua ngưỡng là **80,64% / 73,49%**, **68,79% / 57,74%** và **74,00% / 65,86%**. Số ứng viên vượt ngưỡng khác nhau giữa hai nhánh. Bảng số này đo trường hợp được nâng qua một mức điểm cố định.

## Đối chứng cùng số ảnh mới

Với mỗi truy vấn, lấy những ảnh mới vào top-K khi so với top-K của Dot. So cặp hai nhánh bằng số ảnh mới nhỏ hơn trong hai nhánh, lấy theo thứ tự của từng nhánh. Phép chấm lại tái lập chính xác precision gốc trên từng truy vấn trước khi đo. Các tỷ lệ sau tính trên những ảnh mới đã ghép số lượng. [Chi tiết số cặp, khoảng bootstrap và phân tích theo posterior](mechanism-audit.json)

| Phép so | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin |
|---|---:|---:|---:|
| Hướng thật / hướng tráo | 79,53 / 78,60 | 70,11 / 67,04 | 74,43 / 73,63 |
| Từng từ / sau Dot | 83,79 / 83,79 | 72,57 / 72,62 | 79,66 / 79,76 |

Lợi ích của hướng thật khi lấy trung bình theo lớp là **+0,78 / +3,55 / +1,26** điểm. Khoảng bootstrap theo lớp nằm trên 0 ở cả ba split. Với đối chứng sau Dot, chênh lệch trung bình theo lớp là **+0,12 / −0,02 / −0,05** điểm, và cả ba khoảng đều chứa 0. Chia truy vấn theo bốn mức \(1-\max_w q_w\) trong từng lớp cũng chưa cho thấy nhóm posterior phân tán hưởng lợi hơn từ việc xử lý từng từ.

Kết quả ủng hộ việc kiểm tra sự phân bố của bằng chứng theo các hướng lân cận. Giả thuyết rằng việc chứng nhận mỗi từ riêng giúp thêm khi hình học đã cố định chưa được xác nhận. Mỗi từ hiện có thể lấy một tập nhân chứng khác nhau, nên tổng sau cùng vẫn có thể ghép các phần bằng chứng mà chưa có một nhóm ảnh nào cùng thể hiện. Đây là một giới hạn của công thức đã chấm. Cùng cấu hình cho TU-Berlin kiểm tra chuyển bộ dữ liệu trong chiều sketch → photo.

Thời gian dựng graph, tạo cả các bảng từ đã đối chứng và kiểm tra kernel sau khi có feature và posterior là 0,83 / 0,70 / 1,15 giây. Chấm online của cấu hình từng từ chỉ dùng một tích ma trận 601 chiều như Dot. Phần lớn thời gian của lượt thử đến từ phát lại hàm KD-SBIR gốc trên CPU.
