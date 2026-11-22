# Có thể bảo toàn nội dung của patch khi can thiệp vào neuron nhạy với cách thể hiện ảnh không?

Một thí nghiệm trên CLIP ViT-B/32 cho thấy vài neuron MLP mang phần lớn phản ứng chung khi đảo dấu một thay đổi nhỏ của ảnh. Trên 240 ảnh và 240 sketch độc lập, tám neuron của block 2 nhận diện từ một nhóm ảnh khác chiếm trung vị trên 92% tổng đóng góp tuyệt đối vào phản ứng do phép đo chi tiết patch đó tạo ra. Phản ứng đo được có thể trùng với tín hiệu thị giác hữu ích. Câu hỏi mới là điều gì xảy ra khi thay các giá trị này bằng giá trị thường thấy ở vùng ảnh đích có ngữ nghĩa gần, thay vì chọn neuron rồi triệt tiêu mọi hoạt hóa. [Thí nghiệm nhận diện neuron](https://github.com/HieuNM1804/training-free-ZS-SBIR/tree/bfc769d/docs/results/patch-response-mediation)

## Bộ nhớ chỉ dùng ảnh đích không nhãn

Với mỗi split, lấy ngẫu nhiên ba nhóm ảnh tách biệt từ chính gallery, dùng seed 40483. Dùng 192 ảnh để nhận diện neuron, 768 ảnh để tạo bộ nhớ và 144 ảnh để chẩn đoán độ chính xác của phép thay. Việc lấy mẫu thứ tự tệp được xác định trước khi chấm điểm, không cần tên lớp và không dùng sketch.

Ở bước nhận diện, dùng hai phép dịch pixel có dấu đối nhau trên từng trục, hoặc hai lần đo với nhiễu độ sáng độc lập biên độ 2/255 ở tọa độ RGB. Tại MLP của block 2 hoặc 3, tính phần tăng thêm khi truyền riêng hai quan sát rồi lấy trung bình, so với truyền trạng thái ở giữa. Dùng trọng số đầu ra thật để đo đóng góp của từng neuron. Chọn 8 hoặc 32 neuron mang tỷ lệ đóng góp tuyệt đối lớn nhất trên ảnh calibration. Các phép biến đổi này chỉ chạy khi lập bảng gallery. Hướng đã phát hiện bằng không gian vô hình của tokenizer là một đối chứng giải thích trước đó. Mã của phương pháp mới không cần thực hiện SVD hay trích xuất contour để lập bảng.

Với mỗi ảnh gallery trong bộ nhớ, mã hóa 49 token một lần. Ở mỗi vị trí, lưu giá trị thật của các neuron đã chọn, đặc trưng patch tầng cuối sau layer norm và phép chiếu CLIP, đặc trưng patch đầu mạng và đặc trưng toàn ảnh. Có 37.632 patch trong bộ nhớ của một split. Toàn bộ ảnh của split vẫn là ứng viên xếp hạng.

## Can thiệp và điều kiện bác bỏ

1. Cho một sketch mới, chạy CLIP nguyên thủy để lấy đặc trưng và hoạt hóa ban đầu. Ở từng patch, tìm 4 hoặc 32 patch ảnh đích gần nhất theo cosine ở tầng cuối.
2. Thay các tọa độ neuron đã chọn bằng giá trị trung bình quan sát trên các patch đó. Đặt hiệu giữa mới và cũ qua đúng ma trận đầu ra của MLP và tiếp tục phần còn lại của encoder. Token CLS, những tọa độ khác và trọng số vẫn giữ nguyên. Đây là phép sửa trên ảnh gốc, khác với lấy trung bình nhiều phiên bản ảnh.
3. Đo thay đổi điểm qua từ công khai, cosine đối với ảnh gallery hoặc cả hai. Đánh giá trên nền vùng và diffusion RGB hiện có, đồng thời xuất điểm cosine độc lập trước và sau can thiệp để đo riêng chất lượng đặc trưng. Mỗi truy vấn độc lập.
4. Cho từng đối chứng chọn cấu hình riêng. Thay bằng 0, trung bình gallery, tráo quan hệ khóa–giá trị trong bộ nhớ, chọn patch theo tầng sớm, chọn ảnh theo CLS, chọn neuron ngẫu nhiên, chọn neuron có hoạt hóa tự nhiên lớn và chọn neuron khác khớp độ lớn. Kiểm tra xem từ patch có tái dựng tốt hoạt hóa bản gốc trên ảnh giữ lại hơn dùng trung bình toàn gallery hay không. Một lợi ích tương đương với tráo khóa–giá trị làm yếu lập luận rằng việc giữ nội dung patch là nguyên nhân tăng hạng.

Hai pilot lấy 36 sketch mỗi lớp với toàn gallery. Chọn theo trung bình mAP hai split Sketchy unseen, có ưu tiên precision khi hòa; đây là thiết lập thăm dò. Cấu hình, nhóm đối chứng và băm được lưu trước khi chấm TU đầy đủ. So sánh toàn tập dùng đúng evaluator KD-SBIR, gồm mAP@all/P@100 cho Ext-1 và TU, mAP@200/P@200 cho Ext-2. Cả hai benchmark đều kiểm tra sketch → photo.

## Phạm vi đóng góp

Zintgraf và cộng sự thay một phần đầu vào bằng mẫu có điều kiện để đo ảnh hưởng tới dự đoán của CNN. Ở đây, đối tượng cần thay là phản ứng của một nhóm neuron được phát hiện từ phép đo của chính CLIP. Phân phối điều kiện đến từ patch thuộc miền đích và đầu ra cần kiểm chứng là thứ hạng xuyên miền. Jiang và cộng sự là công trình liên quan quan trọng về neuron của register, với tiêu chí nhận diện và phép sửa khác. Rút ra kết luận đối với register cần đo trực tiếp độ trùng giữa hai nhóm neuron. [Zintgraf và cộng sự, ICLR 2017](https://arxiv.org/abs/1702.04595) · [Jiang và cộng sự, NeurIPS 2025](https://avdravid.github.io/test-time-registers/)

Ba nguồn cơ chế trực tiếp cho hệ thống đầy đủ là phép thay có điều kiện, nền truy hồi qua từ công khai và diffusion trên gallery ảnh. Đặc trưng cosine độc lập chỉ dùng phần can thiệp. [FreeDom, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Efthymiadis_Composed_Image_Retrieval_for_Training-Free_Domain_Conversion_WACV_2025_paper.html) · [Wang và cộng sự, CVPR 2012](https://pages.ucsd.edu/~ztu/publication/cvpr12_CrDP.pdf)
