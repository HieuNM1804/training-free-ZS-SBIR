# Tín hiệu truy hồi còn lại sau khi trừ phản ứng của CLIP với một phần ảnh

Một vector đơn giản giữ được phần lớn lợi ích của nhánh thuần thị giác. Mã hóa toàn ảnh thành q, chia 49 token gốc thành sáu nhóm theo độ giống nhau trong embedding patch của CLIP và lần lượt cho mô hình thấy một nhóm. Những vị trí còn lại nhận token trung bình của chính ảnh. Gọi f là trung bình sáu embedding CLIP đã chuẩn hóa. Chấm gallery bằng q − 0,8f rồi truyền bằng chứng trên đồ thị ảnh RGB. Nếu p là embedding của ảnh gallery, thì điểm trước đồ thị là (q − 0,8f)·p. Ở cấu hình đã khóa, bước sau dùng G(exp(64 × điểm)). G là cùng một toán tử trên gallery cho mọi truy vấn. [Phương pháp và các đối chứng](../../QUERY_FRAGMENT_REFERENCE.md)

## Kết quả đầy đủ trên Colab L4

Mỗi ô là mAP / precision theo phần trăm. Ext-1 dùng mAP@all / P@100, Ext-2 dùng mAP@200 / P@200, TU-Berlin dùng mAP@all / P@100. Các split có lần lượt 14.179 / 17.100, 12.694 / 12.552 và 2.130 / 27.262 truy vấn / ảnh gallery. TU-Berlin có đủ 30 lớp.

| Phương pháp | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin |
|---|---:|---:|---:|
| Cosine CLIP ViT-B/32 gốc | 30,29 / 41,64 | 36,34 / 33,85 | 29,07 / 46,27 |
| Truyền điểm cosine trên gallery RGB | 50,31 / 41,93 | 38,88 / 41,19 | 48,93 / 45,42 |
| Công thức một phía đã thăm dò trên T4, đọc trực tiếp | 57,00 / 71,48 | 65,02 / 59,76 | 55,35 / 69,18 |
| Bổ sung phép xáo vị trí chung ở ảnh đủ và các mảnh | 68,93 / 73,12 | 67,05 / 64,00 | 72,44 / 77,25 |
| Màu trung bình với cấu hình đồ thị riêng đã chọn trên Sketchy | 70,06 / 73,88 | 70,31 / 65,71 | 62,39 / 66,52 |
| **Sáu mảnh, hệ số 0,8 và đồ thị RGB** | **74,53 / 75,81** | **71,96 / 68,86** | **73,27 / 74,21** |
| Nền riêng dùng từ công khai, vùng ảnh và đồ thị | 84,37 / 83,85 | 77,01 / 74,84 | 80,07 / 80,92 |
| Nền từ công khai thêm điểm thị giác tốt nhất trên Sketchy | 84,35 / 84,06 | 77,27 / 75,11 | 79,95 / 80,65 |

Nhánh dùng từ công khai chọn một cách gộp khác công thức thuần thị giác. Nó so sánh hỗ trợ của ảnh đủ và hỗ trợ trung bình của các mảnh sau khi cả hai đã đi qua đồ thị, rồi cộng với hệ số 0,08 vào điểm nền. Chênh lệch mAP so với nền là −0,02 / +0,26 / −0,12 điểm. Cả ba khoảng tin cậy theo lớp đều chứa 0. Cấu hình chính bổ sung phép xáo bố cục cho mAP thấp hơn cách dùng sáu mảnh trực tiếp ở cả ba split. Trên TU, nó có P@100 cao hơn.

Hai split Sketchy được dùng để thăm dò 654 cấu hình trên pilot và so sánh danh sách rút gọn 151 cấu hình trên toàn bộ truy vấn. Chọn theo trung bình mAP hai split với trọng số bằng nhau, dùng precision để phân hạng khi hòa. Cấu hình và 25 hàng phát lại đã được lưu trong Git trước lần chấm TU-Berlin, dùng cùng checkpoint OpenAI ViT-B/32, không huấn luyện, mỗi truy vấn được xử lý độc lập. Sai số lớn nhất giữa hai cách tính metric là 2,39 × 10⁻⁷. Các báo cáo, tham số, rank đầu và số liệu theo truy vấn đi kèm SHA-256 trong thư mục này. [Commit khóa cấu hình](https://github.com/HieuNM1804/training-free-ZS-SBIR/commit/0379d5164c3e7fec61f94fe36e082b57d67b554a)

## Đối chứng cùng hệ số, nhiệt độ và đồ thị

Giữ hệ số 0,8 và nhiệt độ 64 cho cả năm hàng. Đồ thị, gallery và query giống nhau. Sự khác nhau duy nhất là vector được trừ khỏi embedding truy vấn. Các phép đọc này dùng thứ hạng toàn gallery và evaluator KD-SBIR gốc.

| Vector tham chiếu | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin |
|---|---:|---:|---:|
| Ảnh phẳng mang màu trung bình của truy vấn | 71,18 / 71,79 | 67,93 / 64,74 | 65,71 / 65,32 |
| Ảnh có cùng token trung bình tại mọi vị trí | 70,04 / 69,28 | 65,61 / 62,67 | 68,08 / 68,04 |
| Mảnh có cùng diện tích, đổi thành viên mỗi nhóm bằng hoán vị | 73,21 / 74,24 | 70,44 / 67,54 | 72,46 / 73,47 |
| Cùng các mảnh, xáo vị trí token trong ảnh | 74,28 / 75,43 | 71,31 / 68,28 | 73,57 / 75,33 |
| Sáu mảnh ở vị trí gốc | 74,53 / 75,81 | 71,96 / 68,86 | 73,27 / 74,21 |

So với ảnh màu trung bình, chênh lệch mAP là +3,35 / +4,02 / +7,56 điểm. Khoảng bootstrap 95% theo lớp trên Ext-1 là +0,74 đến +5,93, trên Ext-2 là −0,01 đến +8,79, và trên TU là +4,83 đến +10,47. So với mảnh cùng diện tích nhưng đổi thành viên, chênh lệch là +1,33 / +1,52 / +0,81 điểm, cả ba khoảng bootstrap theo lớp đều phía trên 0. Giữ thành viên của từng nhóm và chỉ xáo vị trí cho chênh lệch +0,25 / +0,65 / −0,30 điểm khi chuyển về vị trí gốc. Cả ba khoảng tin cậy của so sánh bố cục này chứa 0. Các ước lượng trên Sketchy dùng lại split đã thăm dò.

## Dùng cấu hình thuần ảnh trên một truy vấn mới

`pixels` là batch ảnh chuẩn hóa bằng preprocessing ViT-B/32, `visual` là image encoder OpenAI ViT-B/32 đóng băng, và mỗi hàng trong `gallery` là một embedding ảnh cùng checkpoint đã chuẩn hóa L2 trên GPU. Dựng graph một lần cho mỗi gallery. Phép chuyển mảnh sang FP16 rồi về FP32 khớp độ chính xác của cache trong thí nghiệm.

```python
import torch
from tf_sbir.query_fragment_reference_inputs import encode
from tf_sbir.query_fragment_likelihood import GalleryWalk

with torch.inference_mode():
    graph = GalleryWalk(gallery)
    query, partial, _ = encode(visual, pixels, ["appearance"])
    reference = partial["appearance"].half().float().mean(1)
    scores = graph.log(64 * ((query - 0.8 * reference) @ gallery.T))
    ranking = scores.argsort(dim=1, descending=True, stable=True)
```

## Các lỗi truy hồi có thể kiểm tra

Khi thay ảnh chỉ có màu trung bình bằng sáu mảnh thật, giữ nguyên hệ số 0,8, nhiệt độ 64 và đồ thị, số top-1 chuyển từ sai sang đúng / từ đúng sang sai lần lượt là 835 / 270 trên Ext-1, 1.010 / 725 trên Ext-2 và 233 / 37 trên TU. Ở Ext-2, một sketch cửa ra vào từng trả về ảnh cửa sổ, sau hiệu chỉnh trả về đúng cửa ra vào. Cũng trên split này, một sketch hươu cao cổ từng trả về đúng lớp lại chuyển sang tê giác. Ở TU, đồng hồ đeo tay chuyển từ búa về đồng hồ, còn một xe buýt chuyển sang xe tải. Các ví dụ là trường hợp trung vị về thay đổi khoảng cách điểm trong nhóm sửa đúng hoặc làm sai, theo quy tắc cố định.

Giữ nguyên các mảnh và chỉ xáo bố cục là đối chứng khó hơn. Khi chuyển từ mảnh xáo sang vị trí gốc, số top-1 được sửa / làm sai là 213 / 191, 284 / 254 và 27 / 38. Cosine trung bình giữa hai vector tham chiếu nằm trong khoảng 0,988–0,989. Dữ liệu cho thấy tham chiếu theo từng ảnh chứa thông tin vượt quá nền màu. Bằng chứng về vai trò riêng của bố cục yếu hơn. Lệnh dưới đây dựng lại hình minh họa trên Colab từ các ID đã lưu và ảnh benchmark đang có.

```bash
python -m tf_sbir.query_fragment_examples --output /content/sbir/results/query-fragment-reference/rank-examples.png
```

## Đóng góp có thể bảo vệ

MARL đã học vector modality rồi trừ nó trong ZS-SBIR, còn Sain và cộng sự đã dùng patch shuffling khi huấn luyện. Hai công trình đặt ra các tiền lệ gần cho phép trừ vector và xáo patch. Khác biệt có thể nghiên cứu ở đây là đo trực tiếp phản ứng của CLIP với các quan sát thiếu của **chính một truy vấn** khi suy luận, lượng hóa phần tăng hạng vượt nền màu và kiểm tra trên một bộ dữ liệu khác với cùng cấu hình. Hai benchmark cùng đánh giá sketch → photo. Độ tổng quát sang các cặp miền khác và mức novelty phù hợp hội nghị A/A* còn cần thêm bằng chứng. [Lyou và cộng sự, WACV 2024](https://openaccess.thecvf.com/content/WACV2024/papers/Lyou_Modality-Aware_Representation_Learning_for_Zero-Shot_Sketch-Based_Image_Retrieval_WACV_2024_paper.pdf) · [Sain và cộng sự, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Sain_CLIP_for_All_Things_Zero-Shot_Sketch-Based_Image_Retrieval_Fine-Grained_or_CVPR_2023_paper.pdf)

Hai nhánh kế tiếp đã kiểm tra khả năng tự tính hệ số và khả năng bỏ bước chia sáu nhóm. [Phép chiếu theo từng truy vấn, đủ ba split](https://github.com/HieuNM1804/training-free-ZS-SBIR/tree/25c2850a3ba391fbb9ae6d484330e544a5caf2d5/docs/results/query-conditioned-nuisance-projection) · [Đối chứng phân bố patch, hai pilot Sketchy](https://github.com/HieuNM1804/training-free-ZS-SBIR/tree/7f80b80e352b35c7e775b2ed03b648122c44e34b/docs/results/patch-marginal-null)
