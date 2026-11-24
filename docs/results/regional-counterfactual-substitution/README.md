# Vùng ảnh đúng lớp có thay được một phần sketch?

Nếu một vùng ảnh bổ sung được nghĩa còn thiếu của sketch, thay vùng sketch bằng vùng ảnh đúng lớp có thể bảo toàn các nghĩa mà sketch gợi ra tốt hơn thay bằng vùng ảnh sai. Kiểm tra trực tiếp khả năng thay thế ấy trong Candidate-Conditioned Regional Matching. Với mỗi cặp sketch–ảnh, lấy hai dải patch ở hai phía, giữ nguyên vị trí và lần lượt cho sketch hoặc ảnh cung cấp mỗi dải. CLIP mã hóa chung bốn tổ hợp. Lặp theo trục ngang và dọc, dùng chung từ điển Open Images 20.931 từ và cùng bộ prompt cho hai miền.

Hai biến nhị phân cho biết nguồn cung cấp hai dải. Biến thứ ba là từ do CLIP dự báo. Để biết sự thay đổi nghĩa nào chỉ xuất hiện khi đổi đồng thời hai nguồn, dựng phân bố entropy cực đại giữ đúng cả ba phân bố hai biến. Phần chênh còn lại đo tương tác bậc ba. Phép thử chính cân phần này theo xác suất từ của sketch nguyên vẹn. Ứng viên gây xáo trộn ít hơn được ưu tiên. Nguyên lý thay thành phần xuất phát từ interchange interventions, còn phép tách tương tác xuất phát từ connected information. Phép thử dùng đầu vào ảnh và token tại vị trí gốc. [Geiger et al., NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/file/4f5c422f4d49a5a807eda27434231040-Paper.pdf) · [Schneidman et al., Physical Review Letters 2003](https://www.weizmann.ac.il/brain-sciences/labs/schneidman/sites/neurobiology.labs.schneidman/files/publications/Schneidman-Still-Berry-Bialek-2003.pdf)

| Tín hiệu quyết định | Ext-1, đúng cân bằng (%) | Ext-2, đúng cân bằng (%) |
|---|---:|---:|
| Tương tác bậc ba, cân theo nghĩa của sketch (chính) | 45,90 | 45,30 |
| Tương tác bậc ba, cân theo bốn tổ hợp | 40,26 | 50,28 |
| Thông tin bậc thấp tái tạo được từ các phân bố hai biến | 60,00 | 53,87 |
| Tổng thông tin | 59,49 | 54,14 |
| Jensen–Shannon giữa hai đầu vào thuần sketch và thuần ảnh | 52,31 | 48,34 |

Kết quả chẩn đoán dùng 850 / 998 cặp so sánh đúng lớp–sai lớp sát hạng, với ảnh đúng khác ảnh mẫu. Trong đó 195 / 362 sketch có đủ cả hai chiều của quyết định tham chiếu để chấm cân bằng. Khoảng bootstrap 95% của chênh lệch so với mức ngẫu nhiên, khi cho mỗi **lớp** trọng số bằng nhau, là [−6,49, +15,67] / [−13,76, +3,23] điểm phần trăm. Điểm trong bảng cho mỗi **truy vấn** đủ điều kiện trọng số bằng nhau. So với tín hiệu bậc thấp, phép thử chính mất 14,10 / 8,56 điểm theo truy vấn.

Mức xáo trộn trung bình của ảnh đúng và sai lần lượt là 0,04130 và 0,04363 nat ở Ext-1, 0,04245 và 0,03940 nat ở Ext-2. Chiều của hiệu ứng đảo giữa hai split. Tương tác giữa hai phần đủ để CLIP đổi nghĩa, nhưng kích thước của nó chưa chỉ ra ảnh nào đúng lớp. Bằng chứng hiện tại bác bỏ việc lấy khả năng thay thế đối xứng ở cùng tọa độ làm tiêu chí truy hồi chính trên tập cặp này. Kết quả phụ thuộc cách chọn cặp thăm dò, gồm những cặp đã được dùng để chẩn đoán trước đó.

Chạy trên Colab Tesla T4, frozen OpenAI CLIP ViT-B/32, seed 42, commit `8d71ea9`. [Protocol đã khóa](protocol.md) · [Kết quả Ext-1](sketchy_1.json) · [Kết quả Ext-2](sketchy_2.json) · [Bản ghi theo cặp và log T4](paired-T4.zip)
