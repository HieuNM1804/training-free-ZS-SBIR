# Một prompt cho mỗi từ boxable trong S

So sánh trên cùng 601 từ Open Images boxable. Hàng 2 sketch, 2 photo lấy số từng truy vấn từ lượt Colab T4 trước. Hai lượt mới dùng cùng feature ảnh, nhiệt độ, softmax và tích vô hướng. Mỗi sketch vẫn dùng hai góc nhìn gốc và lật ngang. [Đối chứng](../semantic-open-images-vocabulary/README.md) · [Cấu hình đã khóa](freeze.json)

- Với một prompt sketch, lấy embedding CLIP chuẩn hóa của `a sketch of the {t}.`
- Với một prompt photo, lấy embedding CLIP chuẩn hóa của `a photo of the {t}.`
- Với hai prompt, dùng embedding trung bình đã chuẩn hóa của nhánh S gốc.

| S trên 601 từ | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin Ext (30 lớp) |
|---|---:|---:|---:|
| 2 sketch, 2 photo | 75,48 / 77,10 | 72,57 / 71,08 | 74,80 / 75,16 |
| 1 sketch, 2 photo | 74,48 / 76,18 | 71,56 / 69,48 | 74,70 / 75,71 |
| 1 sketch, 1 photo | 74,42 / 76,62 | 70,92 / 68,83 | 73,90 / 75,07 |

Mỗi ô là mAP / precision (%). KD-SBIR dùng mAP@all / P@100 cho Ext-1 và TU-Berlin, mAP@200 / P@200 cho Ext-2. Chấm đủ 14.179 / 12.694 / 2.130 sketch tương ứng trên 17.100 / 12.552 / 27.262 ảnh gallery.

So với hai prompt mỗi miền, mAP đổi −1,00 / −1,01 / −0,10 điểm khi rút xuống một prompt sketch. Precision TU-Berlin tăng +0,55 điểm ở cấu hình này. Khi một prompt mỗi miền, mAP đổi −1,06 / −1,65 / −0,90 điểm.

Chạy trên Colab Tesla T4 với CLIP ViT-B/32 và seed 42. Commit `d99b6aa`.

```sh
PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 \
  python -u -m tf_sbir.semantic_boxable_single_prompt
```

[Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU-Berlin](tuberlin.json) · [Điểm từng truy vấn](paired-T4.zip)
