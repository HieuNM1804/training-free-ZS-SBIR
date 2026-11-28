# Một lớp gây nhầm có giải thích đủ phần tăng của 64-NN?

Trên **8.803 / 14.179 sketch ở Ext-1 (62,08%)** và **3.277 / 12.694 ở Ext-2 (25,82%)**, AP thực đo của 64-NN cao hơn mức tốt nhất có thể đạt bằng cách chỉ hạ hạng ảnh của một lớp sai từ thứ hạng Semantic Dot. Điều này vẫn đúng khi phép chẩn đoán được biết nhãn của toàn gallery, được chọn riêng lớp sai cho từng sketch, rồi đặt lại những ảnh ấy ở vị trí tối ưu. Trên Ext-1, ngay cả khi chọn đối thủ chung tốt nhất cho từng lớp thật bằng toàn bộ nhãn thăm dò, mức tăng tối đa chỉ là **3,06 điểm mAP**. Phương pháp muốn tăng 4–5 điểm ở cả hai split cần vượt hiệu ứng của việc sửa hoàn hảo một nhầm lẫn cố định theo lớp trên Ext-1.

## Toàn bộ benchmark unseen

Chạy đủ 14.179 sketch × 17.100 ảnh, 25 lớp ở Ext-1 và 12.694 sketch × 12.552 ảnh, 21 lớp ở Ext-2. Hai hàng đầu là hệ thống truy hồi đã có, có thể xếp hạng từng sketch độc lập. Mọi hàng còn lại là phép chẩn đoán **có nhãn**. Hạ ảnh của lớp sai được chọn, giữ nguyên thứ tự giữa tất cả ảnh còn lại. Precision trong bảng luôn thuộc chính thứ hạng đạt AP báo cáo. Các phép thử có đối thủ được chọn theo nửa vật thể dùng ảnh nguồn rời nhau. Cả hai nửa chấm cùng gallery. [Thiết kế và chứng minh](../../one-rival-ranking-ceiling.md)

| Phép đo | Ext-1 mAP / P@100 | Δ mAP | Ext-2 mAP@200 / P@200 | Δ mAP |
|---|---:|---:|---:|---:|
| Semantic Dot, 601 tên | 75,35 / 77,10 | — | 72,66 / 71,08 | — |
| Semantic Dot với thông tin từ 64 ảnh lân cận trong gallery | 80,15 / 79,94 | +4,80 | 75,71 / 73,60 | +3,05 |
| Sửa đối thủ mà sketch hay nhầm trong phép đo RGB trên nửa vật thể khác | 76,45 / 78,18 | +1,11 | 77,11 / 73,80 | +4,44 |
| Chọn một đối thủ cho mỗi lớp thật theo AP tối đa của nửa vật thể khác | 78,36 / 80,36 | +3,01 | 80,79 / 76,58 | +8,13 |
| Chọn đối thủ tốt nhất cho mỗi lớp thật trên toàn bộ truy vấn chấm | 78,40 / 80,40 | +3,06 | 80,86 / 76,63 | +8,19 |
| Sửa lớp của ảnh sai đứng cao nhất trong danh sách của từng sketch | 78,79 / 81,46 | +3,45 | 81,91 / 76,78 | +9,25 |
| Chọn đối thủ tốt nhất riêng cho từng sketch | 80,49 / 82,95 | +5,14 | 83,94 / 78,52 | +11,28 |
| Đưa toàn bộ ảnh của cùng đối thủ ở hàng trên xuống cuối | 80,49 / 82,95 | +5,14 | 83,05 / 81,64 | +10,38 |

Trong các hàng chẩn đoán, lớp thật và lớp sai xác định cả **những ảnh nào** được phép di chuyển. Các cận này hào phóng hơn thông tin một sketch có thể truy cập khi triển khai. Trường hợp chọn một đối thủ cho cả lớp trên chính toàn bộ truy vấn còn được tối ưu cả ở bước chọn lớp sai. So giữa hai split, khả năng sửa một lớp đối thủ có trần rất khác. [Theo từng truy vấn, Ext-1](one-rival-summary-sketchy_1.json) · [Ext-2](one-rival-summary-sketchy_2.json) · [Đối thủ chung, Ext-1](shared-rival-sketchy_1.json) · [Ext-2](shared-rival-sketchy_2.json)

## Kiểm tra với mức tăng thật của lân cận gallery

Trong nhóm 64-NN thực sự tăng ít nhất 4 điểm AP so với Semantic Dot, **3.095 / 3.838** sketch ở Ext-1 và **2.217 / 2.999** ở Ext-2 vượt cận một lớp sai. Khi so một sketch với cận riêng của chính nó, khoảng chênh phải lớn hơn 0,000002 AP mới được đếm. Kết quả giới hạn họ phép biến đổi thứ hạng đã nêu. Chủ thể của phép thử là AP của từng truy vấn. Tỷ lệ trên không phải là tỷ lệ đóng góp của từng thành phần trong 64-NN.

Hai ví dụ sau được chọn theo **trung vị** của phần AP vượt cận trong nhóm vừa nêu. Khi khoảng cách đến trung vị bằng nhau, dùng ID sketch nhỏ hơn. Với bản vẽ *camel* ở Ext-1 (`n02437136_9933-5.png`, ID 12953), AP gốc là 90,13%. Ảnh sai đầu tiên thuộc lớp *windmill*. Ngay cả chọn *horse* và hạ hạng hoàn hảo mọi ảnh ngựa cũng chỉ đạt 93,45%. 64-NN đạt 99,64%. Với bản vẽ *mouse* ở Ext-2 (`n02330245_10071-2.png`, ID 18683), đối thủ tối ưu là *raccoon*. Ba mức AP lần lượt là 85,84%, 92,49% và 100,00%. [Các thứ hạng được chấm và ID truy vấn](one-rival-L4.zip) · [Cờ kiểm chứng từng truy vấn](shared-rival-L4.zip)

## Vì sao Ext-2 cần tối ưu cả vị trí hạ hạng?

Gọi rⱼ là hạng ban đầu của ảnh đúng thứ j. Trừ đi số ảnh thuộc lớp sai đã chọn đứng trước nó cho vị trí sớm nhất eⱼ. Tính AP toàn gallery bằng các vị trí eⱼ cho cận đạt được trên Ext-1. Trên Ext-2, hàm KD-SBIR lấy trung bình precision tại **các ảnh đúng thực sự nằm trong top 200**. Một ảnh đúng mới vào cuối danh sách đôi khi làm AP@200 giảm.

Với h ảnh đúng trong top 200, AP tốt nhất bằng trung bình của j/eⱼ với j từ 1 đến h. Đưa những ảnh của đối thủ nằm trước ảnh đúng thứ h xuống ngay sau ảnh đó đạt đúng giá trị này. Kiểm tra mọi h từ số lần trúng ban đầu đến số cao nhất khi dời toàn bộ ảnh đối thủ ra sau, giữ h cho AP cao nhất rồi chọn precision cao hơn khi hòa. Trên Ext-2 có **3.622** truy vấn mà cách đặt tối ưu cho AP cao hơn so với đưa hết cùng lớp sai xuống cuối. Vì vậy bảng báo cả P@200 và kết quả đưa toàn bộ đối thủ xuống cuối. [Mã dựng cận và thứ hạng đối chứng](../../../tf_sbir/one_rival_ceiling.py)

Kaggle, bốn NVIDIA L4, OpenAI CLIP ViT-B/32 đóng băng, seed 42. Mã tính cận ở `e38c24a`, chọn đối thủ theo lớp và kiểm tra 64-NN ở `074f452`. Hàm KD gốc chấm mọi hàng và mọi truy vấn. Mức AP/P của Semantic Dot và 64-NN trùng từng truy vấn với lần lưu thứ hạng trước. Với những danh sách được dựng để đạt cận, sai số tuyệt đối lớn nhất giữa công thức và hàm gốc dưới 2,26 × 10⁻⁷ trên thang [0, 1]. Chạy vét cạn 3.360 phép dời hợp lệ cho 540 danh sách ngắn tái lập cận với sai số tối đa 1,12 × 10⁻¹⁶. Các phép tính dùng nhãn ở đây là thăm dò trên Sketchy. Chưa có phép kiểm tra chuyển bộ cho một phương pháp mới. [Gói cận theo từng đối thủ, SHA-256 `56f580d8…60db31d`](one-rival-L4.zip) · [Gói đối thủ theo lớp, SHA-256 `82bfa2fa…c6ef5c3`](shared-rival-L4.zip)
