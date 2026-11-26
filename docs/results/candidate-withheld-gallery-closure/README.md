# Truy hồi bằng chu trình RGB khi che điểm của ảnh ứng viên

Khi che điểm ngữ nghĩa của một ảnh, các ảnh lân cận sửa được một số lỗi top-1. Hai đối chứng chưa xác nhận giá trị riêng của điều kiện khép chu trình. Phép tráo đầu mút đạt mAP cao hơn trên cả ba split. Vì vậy, kết quả chưa ủng hộ việc chọn cơ chế này làm đóng góp độc lập. [Công thức và giao thức](../../candidate-withheld-gallery-closure.md)

## Kết quả trên toàn bộ tập unseen

| Phương pháp | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin 30 lớp |
|---|---:|---:|---:|
| Dot đủ 601 từ | 75,35 / 77,10 | 72,66 / 71,08 | 73,46 / 73,98 |
| Chu trình khi che ứng viên | 58,79 / 78,37 | 73,36 / 71,01 | 53,10 / 75,12 |
| Tráo đầu mút | 58,88 / 78,47 | 73,85 / 71,46 | 53,24 / 75,31 |
| Hai láng giềng điểm cao, bỏ điều kiện nối | 63,85 / 78,44 | 73,20 / 70,94 | 58,93 / 75,03 |
| RGB diffusion trên cùng đồ thị | 76,83 / 77,86 | 73,64 / 71,97 | 74,98 / 74,60 |

Mỗi ô ghi mAP / precision (%). Ext-1 và TU-Berlin dùng mAP@all / P@100. Ext-2 dùng mAP@200 / P@200. Ba split lần lượt có 14.179 / 12.694 / 2.130 sketch và 17.100 / 12.552 / 27.262 ảnh. Chấm bằng evaluator KD-SBIR gốc trên Colab T4, commit **6a4b4a8**, seed 42. Đồ thị 16 láng giềng hai chiều và mọi tham số đã khóa trước khi chấm. [Cấu hình](freeze.json) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU-Berlin](tuberlin.json) · [Số liệu từng sketch](paired-T4.zip)

## Đường nối có giúp chọn ảnh mới đúng lớp?

Ở mỗi sketch, lấy cùng số ảnh mới so với top-K của Dot cho hai nhánh đang so sánh. Bảng ghi chênh lệch tỷ lệ đúng lớp giữa chu trình thật và đối chứng, lấy trung bình theo lớp, cùng khoảng bootstrap 95%. Đơn vị là điểm phần trăm.

| Đối chứng | Ext-1 | Ext-2 | TU-Berlin |
|---|---:|---:|---:|
| Tráo đầu mút trong từng nửa danh sách láng giềng | −0,15 [−0,46 đến +0,19] | −1,14 [−1,87 đến −0,53] | −0,26 [−0,59 đến +0,002] |
| Bỏ điều kiện nối | −0,17 [−0,76 đến +0,21] | +0,01 [−0,38 đến +0,34] | +0,01 [−0,26 đến +0,27] |

Tính trên toàn bộ ảnh mới đã ghép số lượng, tỷ lệ đúng lớp của chu trình thật / đầu mút tráo lần lượt là 73,90 / 74,09%, 63,81 / 64,89% và 71,99 / 72,27%. Đối chứng tráo giữ số đường đi, các ảnh trung gian và tập điểm láng giềng, với một hoán vị cố định theo seed 42.

## Lỗi sửa được và ảnh thiếu bằng chứng

| Chẩn đoán top-1 | Ext-1 | Ext-2 | TU-Berlin |
|---|---:|---:|---:|
| Lỗi Dot ngoài khoảng điểm sketch gốc và lật ngang | 2.730 | 2.804 | 380 |
| Trong số trên, chu trình thật sửa được | 363 | 216 | 39 |
| Trong số trên, tráo đầu mút sửa được | 348 | 275 | 40 |
| Trong số trên, bỏ điều kiện nối sửa được | 343 | 203 | 33 |
| Trong số trên, diffusion sửa được | 112 | 85 | 10 |
| Chu trình thật sửa đúng / làm sai trên toàn bộ truy vấn | 1.141 / 281 | 692 / 442 | 91 / 62 |

Điều kiện khép chu trình buộc **4.176 / 2.912 / 8.916 ảnh** nhận điểm xác nhận bằng 0 cho mọi sketch, tương ứng **24,42% / 23,20% / 32,70%** gallery. Chúng đứng sau mọi ứng viên có điểm xác nhận dương, kể cả khi Dot ban đầu rất cao. Đây là một nguyên nhân có thể làm giảm chất lượng thứ hạng ở phần sau danh sách. Ext-1 và TU-Berlin đều tăng P@100 so với Dot trong khi mAP@all giảm lần lượt 16,56 và 20,36 điểm. Ext-2 chỉ đo tới hạng 200.

## Liên hệ với cycle convexity

Ở ngưỡng $t>0$, bỏ ứng viên $i$ rồi lấy tập $S$ gồm các ảnh có điểm Dot ít nhất bằng $t$. Theo *cycle convexity*, $i$ được thêm ở bước đầu nếu hai láng giềng khác nhau của $i$ cùng thuộc một thành phần liên thông của đồ thị cảm sinh trên $S$. Thực nghiệm áp dụng chính điều kiện này với đường nối dài tối đa hai cạnh và lấy ngưỡng lớn nhất thỏa điều kiện làm điểm truy hồi. Điểm khác ở đây là giới hạn độ dài, sử dụng ngưỡng ngữ nghĩa theo truy vấn và đo tác động khi tráo đầu mút. Nguyên lý khép chu trình đã có trong lý thuyết đồ thị. [Araujo et al., DMTCS 2018](https://dmtcs.episciences.org/4456)
