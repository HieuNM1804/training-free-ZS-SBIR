# Dot, cosine và top-K từ của từng sketch

Giữ OpenAI CLIP ViT-B/32 đóng băng, 601 từ Open Images boxable và đúng một prompt mỗi miền. Sketch dùng `a drawing of a {t}.` và photo dùng `a photo of the {t}.` Với sketch, lấy trung bình hai posterior CLIP của ảnh gốc và ảnh lật ngang. Với mỗi photo, tính softmax trên đủ 601 từ.

Chấm dot và cosine trên đủ 601 từ. Với mỗi sketch, xếp từ theo posterior của sketch đó rồi giữ K từ đầu, đưa phần còn lại về 0. Thử K bằng 1, 2, 4, 8, 16, 32, 64, 128, 256 và 512. Khi bằng điểm, thứ tự ban đầu của từ quyết định thứ hạng. Mọi ảnh gallery được chấm cho mỗi sketch.

- **Dot top-K** cộng $q_w p_w$ trên K từ đã chọn. Giữ trọng số từ của CLIP. Việc chia lại cho tổng xác suất K từ chỉ nhân mọi điểm của cùng sketch với một hằng số dương.
- **Cosine top-K** dùng vector sketch có K phần tử giữ lại và vector photo đủ 601 chiều. Chia tích vô hướng cho tích hai chuẩn L2, trong đó chuẩn của photo tính trên toàn bộ từ vựng.
- **Text → RGB top-K** dùng các embedding văn bản theo prompt photo của chính K từ đó, lấy tổng có trọng số từ sketch rồi chuẩn hóa L2. Chấm cosine với embedding CLIP RGB toàn ảnh.

Quét trên đầy đủ Sketchy Ext-1 và Ext-2. Chọn riêng K cho từng phương thức theo trung bình cộng hai mAP, dùng trung bình precision để phá hòa. Khóa ba K trước khi chấm TU-Berlin. Sketchy là kết quả thăm dò do chọn K trên unseen. TU-Berlin dùng cùng cấu hình đã khóa và đủ 30 lớp.

Ext-1 và TU-Berlin dùng mAP@all và P@100. Ext-2 dùng mAP@200 và P@200. Lưới trên GPU dùng đúng công thức xếp hạng và mẫu số AP của KD-SBIR, đối chiếu hàm gốc trên 24 truy vấn được định trước cho mỗi cấu hình. Bảng chốt chấm mọi truy vấn của năm cấu hình bằng hàm KD-SBIR gốc. Seed 42, GPU Colab, và nhãn chỉ đi vào evaluator.

```sh
PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 \
  python -u -m tf_sbir.semantic_boxable_sparse
```
