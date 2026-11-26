# Chuyển trực tiếp embedding sketch bằng hai ngữ cảnh của CLIP

Cấu hình một lớp đạt **5,19 / 7,68 / 4,28 mAP** trên Sketchy Ext-1, Ext-2 và TU-Berlin. So với Semantic Dot trên cùng truy vấn, mức thay đổi là **−70,16 / −64,99 / −69,18 điểm mAP**. Toàn bộ 25, 21 và 30 lớp đều có AP trung bình thấp hơn baseline. Thí nghiệm dừng ở đây.

## Kết quả full unseen

Mỗi ô ghi mAP / precision (%). Ext-1 và TU-Berlin dùng mAP@all / P@100, Ext-2 dùng mAP@200 / P@200. Các hàng thị giác chấm cosine trên toàn gallery. Hai lượt đọc của cùng sketch gồm ảnh gốc và bản lật. Chỉ hàng Semantic Dot dùng 601 danh từ.

| Phương pháp | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin 30 lớp |
|---|---:|---:|---:|
| Semantic Dot 601 từ | 75,35 / 77,10 | 72,66 / 71,07 | 73,46 / 73,98 |
| Cosine thị giác CLIP gốc | 30,22 / 41,52 | 36,27 / 33,87 | 28,85 / 45,92 |
| Cosine sau phép chiếu chung dùng để đảo bộ đọc văn bản | 33,82 / 45,88 | 40,34 / 37,40 | 30,88 / 48,31 |
| Cộng một độ dịch chuyển cố định giữa hai prompt | 37,87 / 55,95 | 50,38 / 44,20 | 31,88 / 50,84 |
| Chuyển trạng thái qua hai ngữ cảnh, một lớp | **5,19 / 5,64** | **7,68 / 5,90** | **4,28 / 4,42** |

Bốn GPU L4 trên Kaggle chia lượt suy luận trên hai split Sketchy. Seed 42, đúng checkpoint OpenAI CLIP ViT-B/32. Evaluator gốc của KD-SBIR chấm đủ 14.179 × 17.100, 12.694 × 12.552 và 2.130 × 27.262 cặp truy vấn–gallery theo từng split. Hàng Semantic Dot lấy từ kết quả L4 đã chấm cùng evaluator. Mã xác nhận hash, toàn bộ ID truy vấn, ID ảnh và thứ tự trước khi đối chiếu. [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU-Berlin](tuberlin.json) · [Kết quả ghép cặp từng truy vấn](paired-L4.zip)

## Giả thuyết và kết quả kiểm chứng

- **Hiện tượng đã đo.** Khi chia ảnh nguồn của mỗi lớp thành hai nửa rời nhau, hướng dịch chuyển trung bình từ sketch đến ảnh gốc còn phụ thuộc vào lớp. Thành phần này chiếm 29,0% và 30,0% bình phương độ dịch chuyển trung bình theo lớp. Hai nửa có cosine trung bình giữa các hướng theo lớp đã trừ trung bình chung là 0,951 và 0,954. Khi dùng tên lớp chỉ để chẩn đoán, hướng chênh lệch giữa hai caption cùng tên có cosine với phần dịch chuyển thị giác theo lớp chỉ 0,012 và 0,037. [Đo trên ảnh nguồn](diagnosis-visual.json) · [Đối chiếu caption theo lớp](diagnosis-caption.json) · [Mã phân tích](../../../tf_sbir/depiction_gap_diagnosis.py)
- **Can thiệp đã thử.** Đổi tọa độ embedding sketch sang trạng thái kết thúc của text encoder, giải ngược động lực của ngữ cảnh `a drawing of an object.`, giải xuôi dưới `a photo of an object.`, rồi trở về không gian ảnh chung để truy hồi. Mỗi truy vấn chỉ cần chính sketch của nó. Đặc điểm riêng được đem kiểm chứng là sử dụng lại các hàm residual đã đóng băng dưới hai prefix như hai trường biến đổi phụ thuộc đầu vào. FlowCIR đã dùng dòng biến đổi có điều kiện trong CIR và học trường này từ dữ liệu. Tính mới của cách hiện thực ở đây chưa đủ cho một contribution khi can thiệp làm mất khả năng phân biệt ảnh. [Công thức và prior art](../../depiction-state-flow.md) · [FlowCIR](https://arxiv.org/abs/2607.02284)
- **Dấu hiệu thất bại.** Sau biến đổi, cosine trung bình của hai sketch khác nhau tăng từ 0,773 lên 0,996 trên Ext-1, từ 0,781 lên 0,996 trên Ext-2 và từ 0,782 lên 0,995 trên TU. Trên hai split Sketchy, cosine với ảnh nguồn tương ứng giảm trung bình 0,534 ở cả hai. Chênh lệch này được tính theo danh tính ảnh gốc để mỗi ảnh có cùng trọng số. Các embedding bị dồn về gần một hướng trong không gian đầu ra. Một ánh xạ có thể đảo gần đúng trong phép kiểm tra số học vẫn có thể làm hỏng thứ hạng cosine. [Đo trên Ext-1](sketchy_1.json) · [Đo trên Ext-2](sketchy_2.json) · [Đo trên TU-Berlin](tuberlin.json)

## Những gì được chọn trước khi nhìn điểm truy hồi

Đăng ký ban đầu thử 1, 3 và 6 lớp cuối. Trên 24 sketch và 24 ảnh gallery Ext-1 lấy ngẫu nhiên, yêu cầu cosine chu trình trung vị ≥ 0,999, phân vị 1% ≥ 0,995 và phân vị 1% cosine khi tăng gấp đôi số bước ≥ 0,995. Cấu hình một lớp đạt ba ngưỡng với bốn bước RK4 trên mỗi residual sublayer. Cấu hình 3 và 6 lớp trượt ngưỡng chu trình dù hướng đi xuôi đã hội tụ. Tại 16 bước, cosine chu trình trung vị lần lượt là 0,902 và 0,037. Điều kiện ban đầu yêu cầu cả ba độ sâu đạt ngưỡng đã thất bại. [Giao thức gốc](protocol.json) · [Một lớp](numerics-1.json) · [Ba lớp](numerics-3.json) · [Sáu lớp](numerics-6.json)

Trước khi sinh bất kỳ điểm truy hồi nào, tôi bổ sung một lượt đánh giá mô tả dành cho các độ sâu trong danh sách ban đầu đạt ngưỡng số học. Chỉ một lớp đủ điều kiện. Cấu hình một lớp, bốn bước được khóa trước TU-Berlin. Không có lựa chọn theo mAP trong lượt này. Hai split Sketchy đã tham gia phân tích và phát triển. TU đo chuyển giữa bộ dữ liệu trong cùng chiều sketch → photo và từng được dùng để đánh giá các nhánh trước của dự án. [Quyết định trước khi chấm](numerical-eligibility.json) · [Cấu hình khóa và hash các bảng](freeze.json)
