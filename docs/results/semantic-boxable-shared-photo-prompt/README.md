# Một prompt photo cho cả sketch và photo

S dùng một bộ embedding văn bản từ `a photo of the {t}.` trong cả hai miền, trên 601 từ boxable đã khóa. Đối chứng trực tiếp dùng `a sketch of the {t}.` cho sketch và `a photo of the {t}.` cho photo. Chênh lệch có ghép cặp đo tác động của việc đổi prompt sketch khi giữ prompt photo. Các hàng đối chứng được lấy từ lượt đã lưu, với cùng thứ tự sketch, ảnh gallery và nhãn. [Lượt đối chứng](../semantic-boxable-single-prompt/README.md) · [Cấu hình đã khóa](freeze.json)

| S trên 601 từ | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin Ext (30 lớp) |
|---|---:|---:|---:|
| Hai prompt mỗi miền | 75,48 / 77,10 | 72,57 / 71,08 | 74,80 / 75,16 |
| Một prompt sketch, hai prompt photo | 74,48 / 76,18 | 71,56 / 69,48 | 74,70 / 75,71 |
| Một prompt riêng cho mỗi miền | 74,42 / 76,62 | 70,92 / 68,83 | 73,90 / 75,07 |
| Một prompt photo dùng chung | 72,15 / 73,99 | 69,06 / 66,94 | 73,43 / 74,05 |

Mỗi ô là mAP / precision (%). KD-SBIR dùng mAP@all / P@100 cho Ext-1 và TU-Berlin, mAP@200 / P@200 cho Ext-2. Chấm đủ 14.179 / 12.694 / 2.130 sketch trên gallery 17.100 / 12.552 / 27.262 ảnh.

So với một prompt riêng mỗi miền, prompt photo dùng chung đổi mAP −2,27 / −1,87 / −0,47 điểm và precision −2,62 / −1,88 / −1,01 điểm. Cả hai lượt lấy trung bình posterior của sketch gốc và lật ngang rồi chấm tích vô hướng với posterior của mỗi photo.

Chạy trên Colab Tesla T4 với CLIP ViT-B/32 và seed 42. Commit `327996b`.

```sh
PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 \
  python -u -m tf_sbir.semantic_boxable_single_prompt --shared-photo
```

[Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU-Berlin](tuberlin.json) · [Điểm từng truy vấn](paired-T4.zip)
