# Tái tạo đặc trưng ảnh đích từ các khái niệm của sketch

Trên mỗi sketch, lấy các từ công khai có điểm CLIP cao nhất rồi ước lượng đồng thời hệ số của chúng để tái tạo embedding sketch. Những hệ số này được dùng để dự báo đặc trưng ảnh đích. Với mỗi từ, ảnh trong một nửa gallery cung cấp phân bố đặc trưng tham chiếu để chấm các ảnh của nửa kia. Scorer hoạt động độc lập với GCWE và nhánh vùng. Chạy trên T4, OpenAI CLIP ViT-B/32 đóng băng, seed 42, 1.536 truy vấn mỗi split và toàn bộ gallery theo KD-SBIR.

| Cách truy hồi | Ext-1 mAP@all / P@100 | Ext-2 mAP@200 / P@200 |
|---|---:|---:|
| Cosine giữa sketch và ảnh | 31,35 / 43,09 | 36,40 / 33,95 |
| Khớp phân bố từ trực tiếp | **75,33 / 78,58** | **71,28 / 70,33** |
| Hệ số tái tạo, 16 từ | 49,29 / 63,37 | 55,35 / 51,91 |
| Hệ số tái tạo, 32 từ (chính) | 41,94 / 58,92 | 50,16 / 45,76 |
| Hệ số tái tạo, 64 từ | 34,62 / 53,10 | 43,96 / 38,41 |
| Softmax, cùng 32 từ và bộ giải mã ảnh | 61,94 / 69,83 | 61,51 / 59,35 |
| Hệ số tái tạo, 32 từ và hiệp phương sai đẳng hướng | 62,12 / 65,99 | 60,20 / 58,27 |

Cấu hình chính mất 33,39 / 21,12 điểm mAP so với khớp từ trực tiếp. Dùng nhiều từ để tái tạo sketch làm cả hai split kém hơn. Các đối chứng cũng cho thấy cả việc thay hệ số softmax bằng hệ số tái tạo và việc dùng hiệp phương sai toàn phần đều giảm mAP trong cấu hình này. Dừng hướng tái tạo một vector ảnh duy nhất ở pilot. Mã `2d32eb6` thực hiện phép giải hệ tuyến tính nhỏ ở CPU của Colab, còn CLIP, giải mã gallery và tính điểm chạy trên GPU. [Protocol và nguồn nghiên cứu](protocol.md) · [Ext-1](sketchy_1-pilot.json) · [Ext-2](sketchy_2-pilot.json) · [AP, precision và ID từng truy vấn](paired-T4.zip)
