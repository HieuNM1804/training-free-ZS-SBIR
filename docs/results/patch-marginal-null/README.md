# Phân bố patch của một truy vấn có giải thích được mức tăng không?

Pilot chạy trên Colab L4 với 900 / 756 truy vấn Sketchy, 36 truy vấn mỗi lớp và toàn gallery của từng split. Mỗi tham chiếu chỉ đọc 49 token từ chính truy vấn. Cùng hệ số 0,8, nhiệt độ 64 và đồ thị RGB cho mọi hàng. Mỗi ô là mAP / precision theo phần trăm. [Giả thuyết, phép lấy mẫu và tiêu chí dừng](../../PATCH_MARGINAL_NULL.md)

| Tham chiếu để trừ khỏi embedding truy vấn | Sketchy Ext-1 | Sketchy Ext-2 |
|---|---:|---:|
| Rút độc lập token cho 49 vị trí, 16 lần | 73,29 / 74,33 | 68,83 / 66,55 |
| Hoán vị giữ nguyên tập 49 token, 16 lần | 73,66 / 74,82 | 68,37 / 65,86 |
| Từng token lặp ở mọi vị trí, lấy kỳ vọng chính xác qua 49 token | 72,25 / 73,46 | 71,13 / 68,13 |
| Trừ embedding của ảnh gồm toàn token trung bình | 70,85 / 70,31 | 66,92 / 64,19 |
| Trừ trung bình embedding của sáu mảnh | 74,89 / 76,56 | 74,44 / 70,88 |

Tham chiếu rút độc lập thấp hơn sáu mảnh 1,60 và 5,61 điểm mAP. Khoảng tin cậy bootstrap 95% theo lớp của chênh lệch trên Ext-2 từ −10,25 đến −1,24 điểm. Cosine giữa ước lượng từ hai nhóm tám lần rút là 0,993 ở cả hai split. Phép hoán vị có cosine giữa hai nửa khoảng 0,996. Đây là độ ổn định của vector. Kết quả chưa chứng minh mọi thứ hạng đã hội tụ theo số mẫu.

Cả ba cách giữ phân bố patch đều cao hơn đối chứng token trung bình ở hệ số 0,8, nhưng đều thấp hơn sáu mảnh về mAP ở cả hai split. Đã dừng phương án rút độc lập tại pilot theo tiêu chí đã lưu. Mọi hàng và đối chứng hệ số 1 đều được phát lại trên evaluator KD-SBIR gốc. Sai số lớn nhất 2,39 × 10⁻⁷. Các ZIP chứa hai metrics và tình trạng top-1 của từng truy vấn.
