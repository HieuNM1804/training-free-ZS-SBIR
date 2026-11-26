# So sánh dot, cosine và chọn từ theo sketch

Mỗi sketch chọn K từ có posterior CLIP cao nhất của chính nó. Cùng checkpoint CLIP ViT-B/32, 601 từ boxable, prompt `a drawing of a {t}.` cho sketch và `a photo of the {t}.` cho photo. Cả ba cách top-K giữ trọng số posterior của từ đã chọn. [Protocol](protocol.md) · [Cấu hình khóa trước TU-Berlin](freeze.json)

| Cấu hình S | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin 30 lớp |
|---|---:|---:|---:|
| Dot · 601 từ | 75,35 / 77,10 | 72,66 / 71,08 | 73,46 / 73,98 |
| Cosine · 601 từ | 72,47 / 73,89 | 67,25 / 66,12 | 70,79 / 70,97 |
| Dot · top 64 | 75,28 / 77,13 | 72,73 / 71,11 | 73,35 / 73,96 |
| Cosine · top 32 | 73,00 / 74,41 | 67,76 / 66,72 | 70,99 / 71,07 |
| Text → RGB · top 4 | 60,12 / 68,77 | 64,81 / 60,77 | 54,85 / 64,65 |

Các ô là mAP / precision (%). Ext-1 và TU-Berlin dùng mAP@all / P@100. Ext-2 dùng mAP@200 / P@200. Bảng chốt chấm từng truy vấn bằng KD-SBIR gốc trên Colab T4, seed 42, commit `1be22b3`. Với mỗi cấu hình, chấm đủ 14.179 / 12.694 / 2.130 sketch lần lượt trên 17.100 / 12.552 / 27.262 ảnh.

So với dot đủ 601 từ, thay đổi mAP theo thứ tự Ext-1 / Ext-2 / TU-Berlin là **−0,06 / +0,07 / −0,10** điểm cho dot top 64 và **−2,88 / −5,41 / −2,67** cho cosine đủ từ. Cắt từ ở sketch và giữ dot duy trì gần nguyên mAP. Cosine top 32 cải thiện nhẹ so với cosine đủ từ trên ba split, với mAP thấp hơn dot. Truy vấn trực tiếp gallery RGB bằng embedding của bốn từ tốt nhất trong lưới text → RGB thấp hơn dot đủ từ **15,23 / 7,85 / 18,60** điểm mAP.

## Độ nhạy với K

mAP trung bình (%) của hai split Sketchy full unseen. Các K được chọn riêng theo từng phương thức trước khi chạy TU-Berlin.

| K | Dot | Cosine | Text → RGB |
|---:|---:|---:|---:|
| 1 | 60,75 | 58,94 | 59,00 |
| 2 | 66,04 | 64,27 | 62,01 |
| 4 | 69,92 | 68,01 | 62,46 |
| 8 | 72,21 | 69,70 | 61,67 |
| 16 | 73,45 | 70,31 | 60,44 |
| 32 | 73,92 | 70,38 | 59,22 |
| 64 | 74,01 | 70,26 | 58,10 |
| 128 | 73,99 | 70,10 | 57,10 |
| 256 | 74,00 | 69,95 | 56,32 |
| 512 | 74,00 | 69,86 | 55,91 |

Các giá trị Sketchy dùng để chọn K nên mang tính thăm dò. Phép tính GPU của lưới sai khác tối đa 1,79 × 10⁻⁷ so với KD-SBIR gốc trên 24 truy vấn đối chiếu của mỗi cấu hình. Dot đủ từ chênh tối đa 0,0000014 điểm giữa bảng chốt và đối chứng đã lưu.

[Bảng đủ 32 cấu hình](sweep.csv) · [Điểm từng truy vấn](paired-T4.zip) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU-Berlin](tuberlin.json)
