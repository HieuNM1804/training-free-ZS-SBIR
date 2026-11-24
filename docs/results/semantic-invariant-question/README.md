# Ảnh giúp đặt câu hỏi nào mà một sketch có thể trả lời?

Hai ảnh cạnh tranh xác định một quyết định nhị phân trên 20.931 từ công khai. Mỗi từ có thể bỏ phiếu cho một trong hai ảnh với xác suất riêng. Chọn quy tắc bỏ phiếu phân biệt hai ảnh tốt nhất, đồng thời buộc từng ảnh có cùng xác suất trả lời dưới prompt “sketch” và “photo”. Sau đó cho phân bố từ của toàn sketch trả lời. Chi phí đảo quyết định là mức chính xác tối thiểu phải hy sinh trên hai ảnh để có một câu hỏi cho sketch hòa phiếu. Đối chứng giữ cùng mục tiêu phân biệt và hai bộ prompt nhưng bỏ ràng buộc bất biến.

| Quyết định trên hai ảnh sát hạng, ảnh đúng khác ảnh nguồn | Ext-1 | Ext-2 |
|---|---:|---:|
| Số sketch / số quyết định | 96 / 192 | 94 / 188 |
| Khớp từ trực tiếp (%) | 48,44 | 50,53 |
| Câu hỏi tối ưu trên hai prompt gộp chung (%) | 36,98 | 47,07 |
| Câu hỏi bất biến theo prompt (%) | 35,94 | 48,14 |
| Hiệu bất biến trừ gộp theo truy vấn (điểm %) | −1,04 | +1,06 |
| Khoảng 95% cho hiệu theo lớp (điểm %) | [−4,42, +4,78] | [−3,06, +5,00] |

Mỗi sketch có đủ một quyết định mà truy hồi tham chiếu đúng và một quyết định mà nó sai. Hai split Sketchy có ảnh và lớp giao nhau. Tất cả số trong bảng là độ chính xác trên cặp đã chọn theo nhãn để chẩn đoán. Suy luận từng sketch dùng nguyên ảnh và từ vựng Open Images. Ràng buộc bất biến làm giảm khả năng phân biệt hai phân bố từ trên chính hai ảnh. Với các cặp sát hạng, nó cũng chưa làm giảm độ lệch khi đọc lại ảnh bằng prompt thứ ba chưa dùng trong lời giải.

### Cùng một ảnh nguồn có nhiều bản vẽ

Mỗi nhóm có 24 cặp ảnh và 96 quyết định trên sketch theo từng split. Với hai nguồn cùng lớp, độ chính xác nhận diện nguồn đạt 56,25% / 51,56% cho câu hỏi bất biến và 56,25% / 48,44% cho đối chứng gộp. Việc đo độ lệch giữa câu trả lời trên sketch và trên ảnh nguồn cho thấy giảm từ 0,5484 xuống 0,5068 ở Ext-1, và từ 0,5324 xuống 0,4881 ở Ext-2 trên thang trả lời từ −1 đến +1. Tuy nhiên, tỷ lệ hai bản vẽ cùng chọn một phía giảm từ 79,17% xuống 75,00% và từ 83,33% xuống 72,92%. Với nguồn và đối thủ khác lớp, câu hỏi bất biến nhận diện đúng 81,25% / 84,38%, còn khớp từ trực tiếp đạt 91,67% / 85,42%. Nguồn ảnh chính xác và một ảnh khác cùng lớp là hai đích đánh giá khác nhau.

### Độ chắc của nghiệm và độ tin cậy khi truy hồi

Trong số sketch có đúng một quyết định đúng và một sai, chi phí đảo quyết định chọn được quyết định đúng ở 12/23 trường hợp Ext-1 và 9/30 Ext-2. Chỉ lấy độ lớn phiếu của một câu hỏi tối ưu chọn đúng 13/23 và 12/30. Chưa có bằng chứng cho việc dùng chi phí đảo để quyết định khi nào nên sửa thứ hạng.

Ở Ext-1, một sketch ô dù được so hai lần với ảnh kéo, mỗi lần là một cặp ảnh khác nhau. Lần sai có chi phí đảo 0,342 điểm phần trăm, lần đúng chỉ 0,069 điểm. Các từ đóng góp nhiều nhất cho phiếu sai gồm *dice*, *diamond (gemstone)* và *star*. Ở Ext-2, một sketch quả lê sai trước ảnh kéo có chi phí 0,347 điểm và đúng trước ảnh kiếm có chi phí 0,013 điểm. Đây là các lỗi có chi phí đảo lớn nhất trong nhóm có một quyết định đúng và một sai. Những từ trên giải thích phép tính của CLIP. Chúng không phải nhãn chi tiết do người vẽ cung cấp.

Mọi lượt chạy dùng Colab Tesla T4, frozen OpenAI CLIP ViT-B/32 và seed 42. Mã bộ giải `7e80f8d` đối chiếu đủ 20.931 từ trên hai nhánh, hai loại tập cặp và hai split. Sai khác chi phí đảo tối đa giữa hai cách giải là 5,64×10⁻¹². Kiểm tra độ tin cậy và truy nguyên từ dùng mã `654b024`. Mẫu là thăm dò. Chưa có bằng chứng để khóa phương pháp này và chấm TU-Berlin. Cơ chế có liên hệ với kiểm định quyết định khi có yếu tố gây nhiễu và với bước kiểm tra ảnh ứng viên trong training-free CIR. [Hardt et al., NeurIPS 2016](https://proceedings.neurips.cc/paper_files/paper/2016/file/6a9659feb1216f14f7384ba499518b38-Paper.pdf) · [WISER, CVPR 2026](https://arxiv.org/html/2602.23029v4)

[Công thức và protocol](protocol.md) · [Chẩn đoán độ tin cậy](confidence-protocol.md) · [Kiểm chứng bộ giải](solver-validation.json) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [Kết quả độ tin cậy](confidence.json) · [Đóng góp từ theo từng ví dụ](examples.json) · [Toàn bộ bản ghi ghép cặp](paired-T4.zip)
