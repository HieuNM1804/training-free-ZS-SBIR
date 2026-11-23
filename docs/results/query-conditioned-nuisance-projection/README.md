# Hệ số hiệu chỉnh theo truy vấn cần bao nhiêu thông tin từ gallery?

CLIP ViT-B/32 đóng băng chấm từng truy vấn độc lập trên Colab L4. Trích q từ toàn ảnh, lấy trung bình f của sáu lượt CLIP chỉ thấy từng nhóm patch gốc. Hệ số β giải thích biến thiên của điểm q trên từng cạnh dẫn giữa hai ảnh gần nhau. Từ đó dùng q − βf cho toàn bộ gallery. Ở hàng đọc trực tiếp chỉ cần tích vô hướng với từng ảnh. Các hàng còn lại truyền điểm qua cùng đồ thị RGB. Cấu hình chính đã được xác định trước pilot. Hệ số cố định 0,85 làm đối chứng được chọn trong 13 giá trị, trên cùng hai pilot Sketchy. [Thiết kế và phương trình](../../QUERY_CONDITIONED_NUISANCE_PROJECTION.md) · [Commit khóa trước TU-Berlin](https://github.com/HieuNM1804/training-free-ZS-SBIR/commit/e7f83bc88f334be0924b366384abfa3e91bc0109)

Mỗi ô là mAP / precision theo phần trăm. Ext-1 có 14.179 truy vấn và 17.100 ảnh, Ext-2 có 12.694 và 12.552, TU-Berlin có 2.130 và 27.262. Ext-1 và TU dùng mAP@all / P@100, Ext-2 dùng mAP@200 / P@200. TU có đủ 30 lớp. Phát lại mọi thứ hạng của các cấu hình cuối bằng evaluator KD-SBIR gốc. Sai số lớn nhất so với phép tính vector hóa là 2,39 × 10⁻⁷.

| Điều kiện | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin |
|---|---:|---:|---:|
| Cosine CLIP gốc | 30,29 / 41,64 | 36,34 / 33,85 | 29,07 / 46,27 |
| Hệ số theo cạnh, đọc trực tiếp | 57,73 / 71,75 | 65,61 / 60,36 | 56,69 / 69,69 |
| Hệ số cố định 0,80, có đồ thị | 74,53 / 75,81 | 71,96 / 68,86 | 73,27 / 74,21 |
| Hệ số cố định 0,85, có đồ thị | 74,81 / 76,30 | 72,31 / 69,15 | 73,36 / 74,39 |
| Hệ số từ toàn gallery, có đồ thị | 74,56 / 76,17 | 71,87 / 69,02 | 73,27 / 74,50 |
| Phép chiếu Euclid của riêng truy vấn, có đồ thị | 74,97 / 76,65 | 72,38 / 69,30 | 73,39 / 74,43 |
| **Hệ số từ cạnh lân cận thật, có đồ thị** | **75,02 / 76,75** | **72,42 / 69,37** | **73,37 / 74,42** |
| Cùng công thức β theo cạnh, thay mảnh bằng ảnh trắng | 71,41 / 72,40 | 69,48 / 66,09 | 66,07 / 66,24 |
| Cùng cách tính, tráo vị trí các mảnh | 74,88 / 76,59 | 72,38 / 69,02 | 73,75 / 75,67 |

## Bằng chứng về cơ chế

Khi chia gallery thành hai nửa rời nhau rồi chỉ giữ cạnh nằm trong từng nửa, tương quan giữa hai ước lượng β là 0,995 / 0,996 / 0,997. Tỷ lệ trọng số cạnh nối hai ảnh cùng lớp, đo bằng nhãn sau xếp hạng, là 97,14% / 96,86% / 92,33%. Các thống kê này xác nhận hệ số ổn định và gallery có cấu trúc lớp. Chúng chưa xác nhận lợi ích xếp hạng của việc ước lượng β từ cạnh.

Chênh lệch mAP của β theo cạnh so với 0,85 cố định là +0,215 / +0,109 / +0,002 điểm phần trăm. Khoảng tin cậy bootstrap theo lớp 95% lần lượt từ −0,326 đến +0,781, từ −0,786 đến +0,889 và từ −0,336 đến +0,314. So với phép chiếu Euclid, chênh lệch là +0,051 / +0,044 / −0,025 điểm. Trên TU, hệ số theo cạnh sửa 20 top-1 và làm sai 28 so với 0,85. Điểm chính đáng giữ là ước lượng f từ chính truy vấn. Việc tính β bằng cấu trúc lân cận có bằng chứng yếu về lợi ích tăng thêm.

MARL trong ZS-SBIR đã học vector modality rồi trừ nó khỏi embedding. QuARI đã học phép chiếu theo truy vấn trong image retrieval. Phần đóng góp có thể khảo sát ở đây là ước lượng tham chiếu trực tiếp từ quan sát thiếu của một truy vấn, đối chiếu bằng lỗi xếp hạng thật và chuyển giữa bộ dữ liệu. Cơ sở hiện tại chưa đủ để nhận tính mới cho phép trừ modality hoặc cho phép chiếu theo truy vấn nói chung. [Lyou và cộng sự, WACV 2024](https://openaccess.thecvf.com/content/WACV2024/papers/Lyou_Modality-Aware_Representation_Learning_for_Zero-Shot_Sketch-Based_Image_Retrieval_WACV_2024_paper.pdf) · [Xing và cộng sự, NeurIPS 2025](https://ericx003.github.io/proj/quari/)

Mỗi JSON cuối lưu kết quả, xác thực evaluator, cặp top-1 sửa và làm sai, khoảng tin cậy theo lớp và SHA-256 của file theo truy vấn trong ZIP cùng tên. Hai split Sketchy là thăm dò trên unseen. TU-Berlin đo chuyển bộ dữ liệu trong cùng chiều sketch → photo.
