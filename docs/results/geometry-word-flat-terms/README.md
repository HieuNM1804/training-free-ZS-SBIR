# Baseline ba thành phần và tương tác GCWE × khớp vùng

Cùng OpenAI CLIP ViT-B/32 đóng băng, seed 42, từ điển 20.931 tên, gallery đầy đủ và evaluator KD-SBIR gốc. Mỗi ô là mAP / precision (%). Ext-1 và TU dùng mAP@all / P@100. Ext-2 dùng mAP@200 / P@200. Có 14.179, 12.694 và 2.130 truy vấn với 17.100, 12.552 và 27.262 ảnh tương ứng. TU gồm 30 lớp. Hai split Sketchy có dữ liệu chồng lặp và đã tham gia phát triển hệ thống. Tham số giữ nguyên khi bỏ thành phần.

| Cấu hình | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin |
|---|---:|---:|---:|
| Đủ ba thành phần | 84,78 / 84,24 | 77,84 / 75,66 | 81,49 / 82,20 |
| Bỏ hạng tử ngữ nghĩa toàn ảnh | 82,01 / 81,38 | 74,40 / 72,53 | 75,60 / 74,76 |
| Bỏ điểm vùng theo ứng viên | 82,73 / 82,16 | 75,37 / 73,89 | 80,70 / 81,23 |
| Bỏ GCWE | 84,63 / 84,08 | 77,55 / 75,33 | 80,57 / 81,47 |
| Chỉ ngữ nghĩa toàn ảnh | 82,70 / 82,27 | 75,15 / 73,59 | 80,11 / 81,37 |
| Chỉ điểm khớp vùng | 80,99 / 80,01 | 72,69 / 71,09 | 69,98 / 68,54 |
| Truy hồi bằng phân bố từ sau GCWE | 82,73 / 82,16 | 75,37 / 73,89 | 80,70 / 81,23 |
| Chỉ xếp theo lượng hiệu chỉnh GCWE | 47,14 / 52,42 | 53,30 / 49,56 | 53,61 / 53,63 |
| Cosine CLIP hai hướng nhìn | 30,22 / 41,52 | 36,27 / 33,86 | 28,85 / 45,92 |

Gọi S là log điểm khớp từ toàn ảnh có diffusion, R là log điểm vùng và G là lượng thay đổi log điểm do cập nhật từ bằng hình học. Điểm đầy đủ bằng 0,65 S + 0,35 R + 0,65 G, đến sai số khi tính cùng biểu thức từ hai bản cache của encoder. Bỏ S vẫn giữ được từ điển mà R và G sử dụng nội tại. G tự nó có thể âm ở ảnh đúng khi evidence hình học làm giảm điểm ban đầu. Truy hồi bằng GCWE nghĩa là dùng phân bố từ sau hiệu chỉnh, tức S + G. Đây là hai phép thử khác nhau. [Mã chấm](../../../tf_sbir/geometry_word_flat.py)

Trên nền S, tương tác giữa G và R theo mAP (hiệu của hai mức cải thiện khi thêm G) là +0,120, +0,068 và +0,344 điểm. Khoảng bootstrap 95% theo lớp của tương tác lần lượt là [−0,322; +0,529], [−0,663; +0,666] và [−0,117; +0,842]. Các ước lượng điểm đều dương nhưng chưa đủ bằng chứng cho tương tác ổn định qua lớp. [Phân tích theo cặp](interactions.json)

Diffusion tăng mAP trung bình ở cả ba split. So với tắt diffusion, khi bật lại có 838, 1.094 và 123 truy vấn mất tất cả ảnh đúng trong top 200, trong khi lần lượt chỉ có 4, 1 và 3 truy vấn tìm lại được ít nhất một ảnh đúng. Các trường hợp mất xuất hiện ở 25/25, 21/21 và 25/30 lớp. Đồng thời diffusion sửa nhiều lỗi top-1 hơn số lỗi nó tạo ra. Cần theo dõi cả độ bao phủ của những truy vấn khó lẫn độ chính xác trung bình. [Theo lớp](coverage-by-class.json)

Cả hai lượt baseline seed 42 dùng cùng cache đặc trưng trên L4 đều trùng đến hai chữ số thập phân; sai lệch metric trung bình lớn nhất dưới 0,00002 điểm phần trăm. Các con số trong bảng ở đây lấy từ lần chấm riêng chín cấu hình. [Hai lần chạy](../geometry-word-flat/baseline-vi.md) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU](tuberlin.json) · [Kết quả từng truy vấn](paired-L4.zip)

Nếu được biết AP thật của từng truy vấn rồi chọn bật hoặc tắt GCWE để tối đa hóa AP, mức trần chỉ là 85,92 / 79,68 / 82,46 mAP. Khoảng cải thiện tối đa từ quyết định nhị phân đó là +1,14 / +1,84 / +0,97 so với luôn bật. Đây là chẩn đoán dùng nhãn sau xếp hạng. Bất kỳ cải tiến GCWE vượt mức trần này đều phải thay đổi thứ hạng bên trong ít nhất một lựa chọn. [Mức trần của từng nhánh](oracle-limits.json)
