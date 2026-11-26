# Chọn một prompt cho mỗi miền trong S

Cặp đạt mAP trung bình Sketchy cao nhất trong 416 cặp là `a doodle of a {t}.` cho sketch và `a photo of the {t}.` cho photo. Mỗi miền dùng đúng một template cho toàn bộ 601 từ Open Images boxable. Cặp thắng đạt **74,08% mAP trung bình** trên hai split Sketchy. [Cấu hình đã khóa](freeze.json)

| Cấu hình S trên 601 từ | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin Ext (30 lớp) |
|---|---:|---:|---:|
| Cặp một prompt mỗi miền đã dùng | 74,42 / 76,62 | 70,92 / 68,83 | 73,90 / 75,07 |
| Cặp thắng trong lưới | 75,37 / 77,05 | 72,79 / 71,03 | 73,62 / 74,83 |
| Đối chứng hai prompt mỗi miền | 75,48 / 77,10 | 72,57 / 71,08 | 74,80 / 75,16 |

Mỗi ô là mAP / precision (%). Ext-1 và TU-Berlin dùng mAP@all / P@100. Ext-2 dùng mAP@200 / P@200. So với cặp một prompt mỗi miền đã dùng, mAP đổi +0,95 / +1,87 / −0,28 điểm theo thứ tự Ext-1, Ext-2 và TU-Berlin. Đối chứng cùng từ vựng lấy từ các lượt Colab T4 trước. [Kết quả đối chứng](../semantic-boxable-single-prompt/README.md)

| Hạng Sketchy | Prompt sketch | Prompt photo | Ext-1 | Ext-2 | mAP trung bình |
|---|---|---|---:|---:|---:|
| 1 | `a doodle of a {t}.` | `a photo of the {t}.` | 75,37 / 77,05 | 72,79 / 71,03 | 74,08 |
| 2 | `a doodle of a {t}.` | `a picture of the {t}.` | 75,49 / 76,78 | 72,55 / 70,94 | 74,02 |
| 3 | `a drawing of a {t}.` | `a photo of the {t}.` | 75,35 / 77,10 | 72,66 / 71,08 | 74,00 |
| 4 | `a drawing of a {t}.` | `a picture of the {t}.` | 75,49 / 76,81 | 72,46 / 70,96 | 73,97 |
| 5 | `a simple sketch of a {t}.` | `a photo of one {t}.` | 75,33 / 76,06 | 72,61 / 71,48 | 73,97 |

Lưới khóa trước khi chấm gồm 26 template sketch và 16 template photo. Một số nhóm mẫu lấy từ prompt công khai của CLIP, cùng các biến thể mô tả dạng nét vẽ và ảnh. Chấm toàn bộ 14.179 sketch và 17.100 photo ở Ext-1, 12.694 sketch và 12.552 photo ở Ext-2 cho mọi cặp. Chọn theo trung bình hai mAP với trọng số bằng nhau, dùng trung bình precision để phá hòa. Vì việc chọn dùng điểm trên unseen Sketchy, đây là kết quả thăm dò. [Danh sách template](protocol.json) · [Mẫu CLIP công khai](https://github.com/openai/CLIP/blob/main/data/prompts.md)

Trong lưới, phép tính theo lô dùng thứ hạng và mẫu số AP của KD-SBIR. Đối chiếu tám truy vấn mỗi cặp với evaluator gốc cho sai số tối đa dưới 0,00002 điểm phần trăm. Hai cặp đầu được chấm lại từng truy vấn bằng hàm KD-SBIR gốc. Sau khi khóa cặp thắng, chấm đầy đủ 2.130 sketch và 27.262 photo thuộc 30 lớp TU-Berlin bằng hàm gốc.

Chạy trên Colab Tesla T4, CLIP ViT-B/32 đóng băng và seed 42. Commit `58e5ea6`.

```sh
PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 \
  python -u -m tf_sbir.semantic_boxable_prompt_search
```

[Kết quả 416 cặp](search.csv) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU-Berlin](tuberlin.json) · [Điểm từng truy vấn của các cặp chấm lại](paired-T4.zip)
