# Kiểm tra nhiều ảnh và nhiều từ có cùng ủng hộ một cộng đồng gallery

Với mỗi sketch, tách bằng chứng trên các tọa độ phổ của gallery. Tính phần năng lượng được lặp lại ở hai ảnh khác nhau và hai từ khác nhau, rồi dùng nó để quyết định mức giữ từng dải phổ.

| mAP / precision (%) | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin 30 lớp |
|---|---:|---:|---:|
| Lọc theo hai điều kiện | 81,86 / 81,08 | 73,19 / 71,94 | 79,25 / 79,67 |
| Cùng graph, cùng hạng, không lọc | 81,52 / 81,16 | 73,85 / 72,38 | 79,42 / 79,74 |
| Phổ cố định được tối ưu riêng | 82,49 / 81,56 | 73,16 / 72,13 | 79,91 / 80,48 |
| Chỉ loại trùng ảnh | 81,58 / 81,17 | 73,79 / 72,35 | 79,42 / 79,76 |
| Chỉ loại trùng từ | 81,84 / 81,08 | 73,23 / 71,97 | 79,24 / 79,65 |

Công thức năng lượng cho kết quả đúng khi đối chiếu bằng vét cạn các cặp nhỏ. Bộ lọc có điều kiện thua phổ cố định đã tối ưu trên Ext-1 và TU. Khi ảnh hạng nhất sai, phần trọng số lexical witness đến từ lớp khác với ảnh này trung bình là 27,3%, 25,7% và 31,9% trên các mẫu có phân tầng. Trên toàn bộ mẫu, các số tương ứng là 10,4%, 13,8% và 14,6%. Chẩn đoán cho thấy nguồn xác nhận có thể nhiễu. Nó chưa xác lập rằng thu hẹp các tọa độ phổ là biện pháp sửa lỗi đúng.

CLIP ViT-B/32 đóng băng. Mỗi cấu hình cuối được chấm bằng hàm KD-SBIR gốc trên 14.179, 12.694 và 2.130 sketch ở Colab L4. Ext-1 và TU dùng mAP@all và P@100. Ext-2 dùng mAP@200 và P@200. Chọn trên hai split Sketchy unseen rồi khóa cấu hình trước khi chấm TU. Ở cả hai benchmark, chiều truy hồi là sketch → photo. [Giao thức](../../CROSSED_SPECTRAL_PROTOCOL.md) · [Khóa](freeze.json) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU](tuberlin.json) · [Từng truy vấn](per-query.zip)
