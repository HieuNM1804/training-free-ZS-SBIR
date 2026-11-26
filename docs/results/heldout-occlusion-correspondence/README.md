# Kiểm chứng dự báo phản hồi CLIP khi che hai vùng

Dùng các phép che đơn để suy ra tương ứng giữa vùng rồi dự báo đáp ứng che kép chỉ thay đổi mAP tối đa 0,04 điểm so với dùng đều 24 cách ghép. Chênh tỷ lệ đúng lớp của những ảnh mới khi lấy cùng số ảnh là −0,01 / −0,07 / +0,04 điểm theo lớp, cả ba khoảng bootstrap đều chứa 0. [Giả thuyết và công thức](../../heldout-occlusion-correspondence.md)

| Cấu hình | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin 30 lớp |
|---|---:|---:|---:|
| Dot 601 từ | 75,35 / 77,10 | 72,66 / 71,08 | 73,46 / 73,98 |
| Dự báo che kép sau khi quan sát che đơn | 71,29 / 73,74 | 68,86 / 67,18 | 69,09 / 70,76 |
| Đối chứng dùng đều các phép gán | 71,28 / 73,72 | 68,89 / 67,22 | 69,06 / 70,69 |
| Ghép theo cùng vị trí | 71,11 / 73,83 | 68,91 / 67,24 | 68,97 / 70,82 |
| Trung bình posterior của sáu lần che kép | 72,27 / 74,63 | 69,93 / 68,23 | 69,73 / 71,22 |
| Che kép dự đoán từ log-odds che đơn | 70,56 / 72,81 | 68,33 / 66,54 | 68,83 / 70,02 |

Dùng CLIP ViT-B/32, seed 42, Colab T4. Chấm đủ 14.179 / 12.694 / 2.130 sketch trên 17.100 / 12.552 / 27.262 ảnh. Mỗi ô là mAP@all / P@100 ở Ext-1 và TU-Berlin, mAP@200 / P@200 ở Ext-2. [Cấu hình](freeze.json) · [Kết quả từng truy vấn](paired-T4.zip)

## Tín hiệu tương ứng và lỗi thật

Trong top-K của Dot, số hoán vị hiệu dụng tính từ entropy sau khi nhìn thấy bốn lần che đơn là 23,80 / 23,77 / 23,87 trên 24. Khi ghép hai ảnh có điểm đối chứng gần nhau và khác nhãn liên quan, tín hiệu hai bậc quan sát đồng ý về phép gán chọn đúng 51,62% / 52,42% / 53,29%, tính hòa bằng nửa điểm. Khoảng bootstrap theo lớp cho độ chính xác vượt 50% chứa 0 trên Ext-2 và TU-Berlin. Việc bỏ ánh xạ đồng nhất không làm thay đổi kết luận này. Ví dụ trên Ext-1, sketch chiếc cốc mã n03063073_2767-4 chuyển top-1 từ chuông sang cốc, song lợi ích trên các ví dụ riêng lẻ chưa chuyển thành lợi ích tổng thể. [Ext-1](sketchy_1-binding.json) · [Ext-2](sketchy_2-binding.json) · [TU-Berlin](tuberlin-binding.json)

## Cận trên cho phép tìm top-K

Sau khi có embedding, cận từ bất đẳng thức sắp xếp theo từng từ cho phép chỉ chấm công thức đầy đủ trên trung bình 3,42% / 5,48% / 2,18% gallery. Tập top-K trùng quét đầy đủ ở 192/192 truy vấn kiểm tra. Toàn bộ thứ tự bên trong top-K trùng ở 191/192 do sai số số học của cách chia batch. Độ trễ trung vị trên T4 của tìm có cận và quét đủ lần lượt là 4,07 / 9,00 ms, 3,08 / 6,67 ms và 5,37 / 14,22 ms. Các phép đo này áp dụng cho chính công thức dự báo đã nêu, với 64 truy vấn seed 42 mỗi bộ và gallery đầy đủ. [Ext-1](sketchy_1-index.json) · [Ext-2](sketchy_2-index.json) · [TU-Berlin](tuberlin-index.json)

Commit mã hóa là e9d9265, chấm chính d00280d, kiểm tra đối sánh 246fc22, thử tìm có cận e6ef1c7. Phép kiểm tra đối sánh dùng lại đúng kích thước batch nhân ma trận của lượt chính để tái lập các trường hợp Dot sát nhau ở ranh giới top-K.
