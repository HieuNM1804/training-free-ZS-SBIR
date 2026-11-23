# Kiểm tra phân phối attention giữa một sketch và ảnh ứng viên

Chạy hai pilot Sketchy trên Colab L4, 900 / 756 sketch và toàn gallery. Trong mỗi truy vấn, xếp lại 64, 192 hoặc 384 ảnh đứng đầu của cùng một hệ thống rồi giữ nguyên toàn bộ phần đuôi. Mỗi hệ số và tầng được chọn theo trung bình mAP của hai pilot unseen. Bảng báo mAP / precision theo %, Ext-1 dùng mAP@all / P@100 và Ext-2 dùng mAP@200 / P@200. 63 cấu hình đã được phát lại toàn bộ thứ hạng bằng KD-SBIR. Sai số lớn nhất của phép cập nhật metric theo prefix là 4,18 × 10⁻⁷. [Giả thuyết và nguồn nghiên cứu](../../COUNTERFACTUAL_ATTENTION_COMPATIBILITY.md)

| Điểm hoặc phép xếp lại | Sketchy Ext-1 | Sketchy Ext-2 |
|---|---:|---:|
| Cosine CLIP gốc | 30,74 / 42,22 | 36,65 / 33,88 |
| Cosine rồi xếp lại qua KL attention hai chiều | 34,18 / 57,03 | 49,94 / 41,40 |
| Cosine rồi xếp lại bằng khoảng cách Euclid giữa CLS query đã hiệu chỉnh | 34,30 / 57,04 | 50,86 / 41,97 |
| Điểm trực tiếp dùng các mảnh của query | 58,52 / 72,24 | 66,72 / 61,13 |
| Nền dùng mảnh rồi thêm KL attention | 58,54 / 72,27 | 67,07 / 61,13 |
| Nền dùng từ, vùng và diffusion | 84,64 / 83,96 | 80,63 / 78,38 |
| Nền từ/vùng rồi thêm KL attention | 84,64 / 83,95 | 80,71 / 78,38 |
| Nền từ/vùng rồi dùng đối chứng quan sát patch đồng đều | 84,64 / 83,99 | 80,89 / 78,38 |

Ở nền cosine, phép đo chính sửa 279 và 185 top-1, đồng thời làm sai 9 và 37 trên hai split. Tuy vậy, khoảng cách Euclid giữa hai CLS query cho mAP cao hơn ở cả hai. Trên mẫu ảnh ngẫu nhiên cố định của mỗi gallery, AUC phân biệt cặp đúng/sai của KL tầng 12 là 0,639 / 0,642, so với 0,677 / 0,683 bằng Euclid. Ở nền từ/vùng, KL chỉ tăng −0,005 / +0,081 điểm mAP. Đối chứng cho trọng số quan sát đồng đều đạt mAP cao hơn.

Bằng chứng phù hợp với việc vector query của ViT chứa tín hiệu truy hồi. Vai trò tăng thêm của phân bố attention bản địa theo từng ảnh chưa được xác nhận. Đã dừng ở hai pilot. Các file JSON và ZIP lưu lưới, đối chứng, cặp tăng/giảm hạng và hai metric theo từng truy vấn. Các thiết lập xếp lại 192 ảnh giữ nguyên P@200 trên Ext-2 do tập thành viên top-200 không đổi.
