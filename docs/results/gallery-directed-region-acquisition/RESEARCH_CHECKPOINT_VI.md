# Khôi phục Colab và đánh giá các cơ chế mới cho ZS-SBIR

Cấu hình dùng làm mốc đọc 13 vùng của một sketch cả ở tư thế gốc lẫn sau khi lật. Mỗi ảnh ứng viên dùng phân bố từ của chính nó để cho trọng số cao hơn cho những vùng sketch có khả năng mô tả ảnh đó. Lượt đầy đủ đạt **84,37 / 83,85**, **77,01 / 74,84** và **80,07 / 80,92**. Khóa trên hai split Sketchy trước khi đo TU-Berlin. Trong nghiên cứu này, trung bình mAP của hai split Sketchy là tiêu chí thăm dò, vì vậy các số Sketchy chịu ảnh hưởng của chọn cấu hình.

| mAP / precision (%) | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin 30 lớp |
|---|---:|---:|---:|
| Từ công khai và diffusion gallery RGB | 81,86 / 81,33 | 74,19 / 72,68 | 79,16 / 80,24 |
| Trung bình đều 13 vùng, hai tư thế | 84,09 / 83,70 | 76,47 / 74,27 | 79,83 / 80,74 |
| Chọn vùng theo ảnh ứng viên, chọn và đọc trong cùng tư thế | **84,37 / 83,85** | **77,01 / 74,84** | **80,07 / 80,92** |
| Chọn ở một tư thế, đọc lại ở tư thế còn lại | 84,25 / 83,72 | 76,87 / 74,69 | 79,91 / 80,67 |
| Thêm nội suy token vào mốc chọn vùng theo ảnh ứng viên | 84,34 / 84,04 | 77,12 / 74,97 | 79,99 / 80,68 |
| Chọn trước đúng một vùng từ các nhóm ảnh đang cạnh tranh | 82,97 / 82,54 | 75,20 / 73,00 | 78,73 / 79,48 |

Ext-1 và TU dùng mAP@all và P@100. Ext-2 dùng mAP@200 và P@200. Tất cả hàng dùng cùng checkpoint OpenAI CLIP ViT-B/32 đóng băng, từ điển công khai 20.931 tên Open Images và query độc lập trên Colab L4. Hàm KD-SBIR gốc chấm đủ 14.179, 12.694 và 2.130 truy vấn trên các gallery lần lượt có 17.100, 12.552 và 27.262 ảnh. Hai dataset đều đo sketch → photo. Phương pháp sử dụng prompt nêu tên miền. [Lượt hai tư thế](https://github.com/HieuNM1804/training-free-ZS-SBIR/blob/824f979/docs/results/cross-view-region-verification/README.md) · [Lượt nội suy token](https://github.com/HieuNM1804/training-free-ZS-SBIR/blob/ba905c1/docs/results/gallery-verified-text-interpolation/README.md) · [Lượt chọn một vùng](README.md)

## Đóng góp của cơ chế so với đối chứng

Việc chọn vùng theo từng ảnh ứng viên, tại cùng số lượt chạy CLIP, cải thiện mAP so với trung bình đều thêm 0,279, 0,540 và 0,244 điểm. Khoảng tin cậy 95% bootstrap theo lớp lần lượt là [0,079; 0,458], [0,151; 0,936] và [−0,003; 0,480] điểm. Các phần tăng 2,51, 2,82 và 0,91 điểm trên mốc chữ + gallery còn bao gồm lợi ích của việc đưa các vùng phóng to vào hệ thống. Bước kiểm tra qua ảnh lật làm giảm mAP trên cả ba split. Điểm cho nội suy token gần như hòa với mốc trên trung bình hai Sketchy và giảm 0,08 điểm mAP trên TU. [So sánh theo từng truy vấn](strong-error-comparison.json)

Khi chỉ đọc thêm một vùng, phương pháp dự báo từ gallery xếp đúng thứ tự lợi ích của hai vùng trong 58,4%, 57,1% và 58,9% các cặp phân biệt được. Tuy nhiên, phần lớn lợi ích trên hai split Sketchy đến từ chọn kích thước vùng. Sau khi giữ kích thước, việc dự báo vị trí thêm 0,29, 0,20 và −0,02 điểm mAP so với ngẫu nhiên. Một vị trí cố định được xác định từ gallery đã đạt 83,13, 75,29 và 78,93 mAP ở cùng ngân sách một vùng. Chưa có cơ sở xem chính sách chọn vị trí là đóng góp đạt hiệu quả ổn định. [Phân rã theo scale và vị trí](locked-error-diagnosis.json)

## Các giả thuyết mới kiểm tra trên mẫu

| Phép thử | Bằng chứng về cơ chế | Kết luận |
|---|---|---|
| Nhiều vùng rời nhau phải được cùng một ảnh gallery giải thích | Khác biệt mAP trung bình so với cho phép ảnh khác nhau giải thích từng vùng là khoảng 0,0005 điểm ở cấu hình tốt nhất. | Dừng ở pilot. [Chi tiết](https://github.com/HieuNM1804/training-free-ZS-SBIR/blob/f6d0915/docs/results/partial-object-gallery-evidence/README.md) |
| Dùng tín hiệu ở độ phân giải gốc để dự báo thông tin sau khi zoom | Cross-entropy dự báo từ giảm từ 4,781 xuống 4,715 và từ 4,712 xuống 4,663 nat. Đối chứng ghép sai ảnh vẫn chọn vùng có mAP cao hơn trên Ext-2. | Dự báo được từ, chưa dự báo ổn định được giá trị retrieval. [Chi tiết](https://github.com/HieuNM1804/training-free-ZS-SBIR/blob/ad69ef3/docs/results/preview-conditioned-region-acquisition/README.md) |
| Chấm ứng viên theo thay đổi ngữ nghĩa mà ảnh của nó dự báo | Trên 145 và 161 truy vấn hạng 1 sai ở mốc, tín hiệu mới thiên về lớp đúng hơn lớp gây nhầm trong 54,5% và 50,9% trường hợp. Đối chứng hoán vị đạt 59,3% và 66,5%. | Dừng ở pilot, giữ cơ chế trừ bằng chứng làm giả thuyết đã bị thử thách. [Chi tiết](https://github.com/HieuNM1804/training-free-ZS-SBIR/blob/39e92d4/docs/results/conditional-zoom-innovation/README.md) |

Các pilot dùng 36 sketch mỗi lớp và toàn gallery. Số pilot không so trực tiếp với các hàng chấm toàn bộ truy vấn. Chưa nhánh nào trong ba pilot đáp ứng đồng thời đối chứng cơ chế và ưu thế thuyết phục trên cả hai split để đưa vào phương pháp chính.

## Mốc nghiên cứu và lỗi cần giải thích

SpLIP dùng cùng ViT-B/32 và huấn luyện prompt trong 60 epoch, báo cáo lần lượt 80,2 / 86,7, 76,4 / 77,3 và 73,1 / 78,2. Mốc hiện tại cao hơn về mAP trên ba split và thấp hơn SpLIP 2,85 điểm P@100 và 2,46 điểm P@200 trên hai split Sketchy. C-DisAmb tại CVPR 2026 có huấn luyện và báo cáo 81,2 mAP@200, 78,9 P@200 trên Sketchy. So với con số đó, mốc hiện tại còn cách 4,19 và 4,06 điểm. Những chênh lệch này mô tả kết quả công bố theo cùng tên metric. Quyền sử dụng gallery và cách chọn tham số trên unseen của nghiên cứu hiện tại cần được công khai khi so sánh. [SpLIP, Bảng 1](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03426.pdf) · [C-DisAmb, Bảng 1](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhou_Modeling_the_Visual_Ambiguity_of_Human_Sketches_CVPR_2026_paper.pdf)

Nếu dùng nhãn sau truy hồi để chọn vùng có AP cao nhất riêng cho từng sketch, mAP đạt 88,68, 84,49 và 84,88. So với mốc mạnh đây là dư địa hậu nghiệm +4,31, +7,48 và +4,80 điểm. Trong 25% sketch có AP thấp nhất ở mốc mạnh, dư địa trung bình tăng lên 16,68, 29,11 và 16,75 điểm. Có cả tình huống vùng khác cứu được lỗi và tình huống mọi vùng đều mơ hồ. Ví dụ một sketch cá đuối có AP từ 5,59% ở CLIP cosine lên 91,92% ở mốc vùng. Với một sketch gấu mèo, AP đi từ 62,00% xuống 0,00% và ngay cả vùng tốt nhất trong phép thử chỉ đạt 2,03%. Phân biệt lỗi thiếu chi tiết với lỗi mà kênh từ làm mất bằng chứng thị giác sẽ có giá trị nghiên cứu cao hơn việc tiếp tục dò trọng số của các cơ chế yếu trên. [Ví dụ thật và đối chiếu với CLIP gốc](strong-error-comparison.json)

## Khi hệ thống theo từ lấn át một kết quả đúng của CLIP gốc

Tôi kiểm tra toàn bộ 29.003 truy vấn thêm một lần, giữ nguyên cả hai thứ hạng và hàm KD-SBIR. Trên Ext-1, Ext-2 và TU, cosine của CLIP gốc đưa đúng lớp lên hạng 1 trong 103/2.320, 383/3.215 và 63/407 lỗi hạng 1 của mốc mạnh. Nếu sau khi biết nhãn được chọn nguyên một trong hai thứ hạng theo AP, mAP chỉ tăng 0,05, 3,04 và 0,13 điểm. Đây là điểm hậu nghiệm của một bộ chọn lý tưởng. Nó định lượng giới hạn của chiến lược chuyển nguyên thứ hạng.

Trên Ext-2 có 1.531 sketch mà AP@200 của CLIP cosine cao hơn mốc ít nhất 5 điểm. Trong nhóm truy vấn được xác định hậu nghiệm này, cosine đạt 25,92 mAP@200 / 24,46 P@200 còn mốc vùng đạt 2,05 / 2,72. Trên các split dùng mAP@all, chỉ 39 sketch Ext-1 và 10 sketch TU đạt cùng điều kiện. So sánh giữa các split chịu ảnh hưởng của cả dữ liệu và ngưỡng metric. Có 155 trường hợp hạng 1 của cosine đúng hươu cao cổ trong khi hệ thống theo từ trả về bò ở Ext-2. TU có 25 trường hợp tương ứng giữa gậy bóng chày và búa. Tên lớp chỉ được đọc sau khi xếp hạng để mô tả lỗi. Chưa có bằng chứng về một tín hiệu không nhãn có thể nhận diện đáng tin những trường hợp này. [Thống kê đầy đủ](native-region-disagreement.json) · [AP, precision và kết quả hạng 1 theo truy vấn](native-region-disagreement.zip)
