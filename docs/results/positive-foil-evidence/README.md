# Bằng chứng dương cho cách hiểu cạnh tranh

## Câu hỏi

Khi sketch không đưa từ nào của ảnh A lên nhóm đầu, việc nó ủng hộ một cách hiểu B khác có cho biết A sai hay không? Ý tưởng thiết kế đối chứng lấy từ các trò chơi nhận diện một đối tượng trong tập lựa chọn. Andreas và Klein nghiên cứu cách một mô tả giúp phân biệt mục tiêu với đối tượng cạnh tranh. Fan và cộng sự khảo sát cách người vẽ thay đổi mức trừu tượng theo các đối tượng được đưa vào ngữ cảnh. Trên Sketchy, người vẽ không được giao chính các cặp A/B do chúng tôi tạo. Phép thử khai thác tương quan trong dữ liệu quan sát. [Andreas và Klein, EMNLP 2016](https://aclanthology.org/D16-1125/) · [Fan và cộng sự, Computational Brain & Behavior](https://cogtoolslab.github.io/pdf/fan_cbb_2019.pdf)

## Kiểm tra ảnh nguồn

Mỗi ảnh có tám từ đứng đầu trong 20.931 prompt photo Open Images. Mỗi sketch có 32 từ đứng đầu khi lấy trung bình hai posterior trên sketch gốc và sketch lật. Trong phép chẩn đoán, từ xuất hiện nghĩa là CLIP đặt nó trong danh sách này. Các từ không phải nhãn do người quan sát gán. Chọn B là ảnh mẫu gần A nhất theo cosine trong số ảnh có tám từ không giao với A. C là một ảnh khác của lớp B, có độ gần thị giác với A khớp gần nhất và cũng có tám từ không giao với A. Ở điều kiện có nhãn, B còn phải khác lớp với A. Điều kiện này chỉ dùng để hỏi câu hỏi nghiên cứu.

Với mỗi nguồn A, B, C, giữ các sketch không có bất kỳ từ top 8 của A trong top 32. Yêu cầu cả ba nguồn còn ít nhất một bản vẽ để so sánh. Mỗi cặp A/B có trọng số bằng nhau. File reference-game.json chứa xác suất xuất hiện ít nhất một từ của B ở ba nguồn, sau đó là tỷ lệ A/B cùng lớp. Ext-1 có 1.478 cặp hợp lệ khi yêu cầu B khác lớp, Ext-2 có 1.389 cặp. Tỷ lệ tương ứng cho các sketch nguồn A/B/C là 6,7/64,1/62,1% và 10,1/54,3/54,8%. Hiệu ứng chỉ trực tiếp áp dụng cho các nguồn đã chọn. Các split có thể chia sẻ ảnh hoặc lớp.

## Kiểm tra truy hồi

Suy luận dùng mỗi sketch độc lập và gallery không nhãn. Điểm điều chỉnh cho một ứng viên A là log của một trừ tổng xác suất sketch dành cho tám từ của B. Điểm này được cộng vào điểm log từ điển hoặc điểm của cấu hình phân cấp đã khóa. Số hạng biểu diễn xác suất không lấy phải một từ của đối thủ trong một lần lấy mẫu. Áp dụng hệ số 1 cho cả hai split. B được chọn trước truy vấn từ gallery bằng độ gần CLIP và điều kiện không giao từ. Đối chứng dùng ảnh đủ điều kiện gần thứ hai. Hai ảnh khác nhau nhưng không bắt buộc khác lớp. Dấu âm của bằng chứng là một giả thuyết thực nghiệm.

File sketchy.json lưu lượt đầu trên cấu hình trước khi trung bình hai hướng ở đầu vào diffusion. File sketchy-verified-base.json dùng cấu hình đã xác nhận trên TU-Berlin và ghi rõ mã nguồn. Kiểm tra trên toàn bộ Sketchy, trên Colab NVIDIA L4. Cách tính theo thứ hạng được đối chiếu với evaluator KD-SBIR gốc trên truy vấn đầu mỗi batch cho từng biến thể. Nhãn của benchmark và liên kết ảnh nguồn chỉ dùng sau khi chấm. Không dùng TU-Berlin để chọn biến thể.

## Lệnh trên Colab

```bash
python -m tf_sbir.positive_foil --dataset sketchy
python -m tf_sbir.foil_reference_game
```

Các đếm trong alternative_audit ghi số lượt có ứng viên thay thế, tổng lỗi của top-1, số lỗi có ứng viên thay thế đúng lớp, số lượt chuyển khi từ A vắng và từ B xuất hiện, số lỗi sửa được và số kết quả đúng bị phá. Nếu trong top 200 không có đối thủ đủ điều kiện thì giữ ảnh cũ. Việc lấy tám từ khác nhau về chính tả không đảm bảo các từ hoặc hình ảnh khác nghĩa.

