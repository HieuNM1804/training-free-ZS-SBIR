# Phần nào của ảnh nguồn có thể xác định từ thứ tự các từ CLIP?

Với từng sketch, xét tất cả phân bố từ giữ nguyên thứ tự CLIP và có hạng từ trung bình nằm giữa hai lần đọc ảnh gốc và ảnh lật. Chỉ đưa ra quyết định chắc chắn theo mô hình nếu mọi phân bố trong tập đều ưu tiên cùng một ảnh. Mỗi sketch được suy luận độc lập. Danh tính ảnh nguồn và các bản vẽ khác của cùng ảnh chỉ dùng khi đánh giá. Sketchy cho phép phân biệt biết đúng lớp và biết đúng ảnh đã được dùng để vẽ. Cả hai ảnh trong cặp cùng lớp đều là đáp án liên quan đối với benchmark SBIR theo lớp.

| Cặp ảnh | Split | Quyết định chắc chắn / số truy vấn | Đúng ảnh nguồn trong số đã chọn | CLIP chọn theo độ tự tin, cùng số truy vấn | Bản vẽ khác cùng nguồn đồng ý với quyết định |
|---|---|---:|---:|---:|---:|
| Cùng lớp | Ext-1 | 785 / 3.379 | 54,65% | 58,85% | 95,03% |
| Cùng lớp | Ext-2 | 644 / 2.990 | 56,83% | 56,06% | 93,84% |
| Khác lớp | Ext-1 | 2.292 / 3.372 | 98,60% | 99,83% | 96,95% |
| Khác lớp | Ext-2 | 1.653 / 2.990 | 98,25% | 98,97% | 96,66% |

Ở 187 cặp cùng lớp Ext-1 có ít nhất một quyết định chắc chắn, 183 cặp chỉ có duy nhất một ảnh từng được chọn. Ext-2 có 151/159 cặp như vậy. Hiện tượng này giải thích vì sao các bản vẽ có thể cho lựa chọn giống nhau rất thường xuyên dù độ chính xác theo ảnh nguồn chỉ xấp xỉ 55%. Hoán đổi nhãn nguồn trong từng cặp, giữ cả tần suất ảnh được chọn và số bản vẽ thực của từng ảnh, cho độ chính xác kỳ vọng 49,33% / 50,24%. Phần vượt mức này là 5,32 / 6,59 điểm. Chọn theo độ tự tin của điểm CLIP gốc cho phần vượt tương ứng 8,15 / 5,93 điểm.

Nếu bỏ ràng buộc hạng trung bình, phân bố đều thuộc mọi tập cho phép và cho điểm bằng nhau với mọi phân bố từ của ảnh. Do đó không thể có quyết định thắng nghiêm ngặt nào. Đây là hệ quả toán học của giả định đã chọn. Bằng chứng thực nghiệm có giá trị hơn nằm ở những gì xảy ra sau khi loại trường hợp suy biến này. Khi đó tính ổn định trước thay đổi xác suất từ vẫn không đảm bảo nhận diện được ảnh nguồn. Chưa có lý do dùng tiêu chuẩn này làm bộ xếp hạng chính.

OpenAI CLIP ViT-B/32, Colab T4, seed 42, 25 và 21 lớp theo split chuẩn. [Dẫn xuất biên chính xác, nguồn và tiêu chí](protocol.md) · [Đối chứng thiên lệch chọn ảnh](pair-preference.json) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [Dữ liệu theo sketch](paired-T4.zip)
