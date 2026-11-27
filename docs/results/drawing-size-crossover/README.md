# Kích thước trình bày và chi tiết ảnh ảnh hưởng thế nào tới truy hồi?

Trên hai split Sketchy, CLIP ưu tiên ảnh đúng lớp hơn khi cùng một bản vẽ đã giảm chi tiết được trình bày ở kích thước lớn hơn. Tác động xuất hiện ở cả bộ đọc thị giác gốc và Semantic Dot. Có thể tách được tác động của phần diện tích trình bày khỏi tác động của việc giảm rồi nội suy độ phân giải. Đây là thí nghiệm chẩn đoán trên hai bộ đọc đã có. [Giao thức xác định trước](../../drawing-size-crossover.md)

## Kết quả full unseen

Mỗi hàng dùng cùng gallery gốc và đủ 14.179 hoặc 12.694 sketch. Số trong mỗi ô là mAP / precision (%). Ext-1 dùng mAP@all / P@100 trên 17.100 ảnh, Ext-2 dùng mAP@200 / P@200 trên 12.552 ảnh. Cả hai thước đo đều phát lại qua hàm gốc của KD-SBIR. Có 2.500 và 2.100 ảnh nguồn.

| Bộ đọc và điều kiện truy vấn | Sketchy Ext-1 | Sketchy Ext-2 |
|---|---:|---:|
| CLIP thị giác, bản gốc | 30,22 / 41,52 | 36,27 / 33,87 |
| CLIP thị giác, hiển thị nhỏ | 26,58 / 35,50 | 29,67 / 29,06 |
| CLIP thị giác, cùng bản thu nhỏ phóng lại | 28,11 / 37,55 | 32,37 / 31,20 |
| Semantic Dot, bản gốc | 75,35 / 77,10 | 72,66 / 71,07 |
| Semantic Dot, hiển thị nhỏ | 72,49 / 73,71 | 68,74 / 67,39 |
| Semantic Dot, cùng bản thu nhỏ phóng lại | 74,13 / 75,81 | 70,62 / 69,23 |

Kích thước dùng để thu nhỏ từng bản vẽ do bản vẽ nhỏ nhất của cùng ảnh nguồn xác định. Đo bán kính RMS từ độ tối liên tục trên nền trắng của tập dữ liệu. Co toàn bộ canvas, giữ tâm nét vẽ và đưa đúng raster đã co vào hai lượt hiển thị. Thông tin về bản vẽ khác chỉ phục vụ can thiệp hồi cứu. Tất cả số liệu trong bảng là kết quả của các điều kiện chẩn đoán.

## So sánh hai bản vẽ cùng một ảnh nguồn

Mỗi nguồn đóng góp một cặp gồm bản vẽ lớn nhất và nhỏ nhất. Ext-1 có 2.499 nguồn với hai kích thước phân biệt được, Ext-2 có 2.100. Tỷ lệ bán kính lớn nhất / nhỏ nhất có trung vị 1,63 và 1,65. Sau thao tác, sai số bán kính tương đối trung vị so với kích thước đích nhỏ hơn 0,02% ở cả hai split, dịch tâm trung vị khoảng 0,4 pixel trên canvas 224 pixel.

| Thành phần chênh lệch AP, trung bình cân bằng theo lớp | CLIP Ext-1 | CLIP Ext-2 | Semantic Ext-1 | Semantic Ext-2 |
|---|---:|---:|---:|---:|
| Bản vẽ lớn gốc trừ bản vẽ nhỏ gốc | +6,71 | +14,12 | +4,56 | +7,64 |
| Bản lớn gốc trừ bản lớn sau khi giảm chi tiết và phóng lại | +3,54 | +6,66 | +1,64 | +2,09 |
| Cùng raster đã giảm chi tiết, hiển thị lớn trừ hiển thị nhỏ | **+2,18** | **+4,47** | **+3,01** | **+3,80** |
| Hai bản vẽ khác nhau với bán kính đã khớp | +0,99 | +2,99 | −0,08 | +1,75 |

Khoảng bootstrap 95% theo lớp cho tác động hiển thị của CLIP gốc lần lượt là [0,91, 3,56] và [1,65, 7,23] điểm AP. Với Semantic Dot, các khoảng là [1,81, 4,28] và [1,66, 6,01]. Sau khi bỏ bốn lớp chung, mức chênh của Semantic Dot vẫn là +3,18 và +3,94 điểm. Với hai bản vẽ khác nhau đã khớp bán kính, khoảng 95% của Semantic trên cả hai split chứa 0. Ba thành phần cộng lại chính xác thành chênh lệch tự nhiên trước khi làm tròn. Hiệu ứng hiển thị có cả ảnh hưởng của nội suy khi đưa raster về lưới đầu vào cuối. Số liệu này chưa xác định layer CLIP gây ra hiệu ứng.

## Ví dụ tê giác và bò

Trên cùng hai ứng viên đã chọn từ lần chẩn đoán trước, biên điểm Semantic của bản vẽ số 4 là −0,064 ban đầu, −0,135 khi hiển thị nhỏ, và +0,093 khi phóng lại chính raster đó. Hai bản vẽ số 1 và số 2 vẫn nghiêng về tê giác trong cả ba điều kiện. Điều kiện đủ để đảo một quyết định ở bản vẽ này chưa giải thích toàn bộ khác biệt giữa những nét vẽ tự nhiên. [Hai ảnh ứng viên, các bản vẽ và quy tắc chọn ví dụ](https://github.com/HieuNM1804/training-free-ZS-SBIR/blob/be39f9e59f229912435c37eb306d9e2aa1b11eec/docs/results/native-block-depiction/redraw-example.json)

Toàn bộ lượt chấm chạy trên Kaggle bằng bốn NVIDIA L4, CLIP ViT-B/32 OpenAI đóng băng, seed 42. Giữ nguyên template, trọng số và evaluator. Không có lần dò kích thước để tối ưu mAP. Cosine nhỏ nhất khi phát lại ảnh có kích thước không đổi là 0,9999939. Nhánh dừng ở kết quả chẩn đoán theo yêu cầu của người dùng. [Số liệu chính xác và manifest](summary.json) · [AP, precision và hình học từng truy vấn](per-query-L4.zip)
