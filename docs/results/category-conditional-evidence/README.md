# Hai dấu hiệu đến từ cùng ảnh hay hai ảnh cùng lớp?

Trên hai gallery đầy đủ, phép đo cho biết liệu nhận biết chính xác lớp của từng ảnh có giúp tận dụng sự bổ sung giữa hai ví dụ riêng biệt. Việc dùng nhãn là chẩn đoán hậu nghiệm. Các tỷ lệ dưới đây là phần trăm truy vấn lỗi mà lớp đúng thắng lớp của ảnh top-1 sai sau khi gộp bằng chứng. Chúng có cách tính khác mAP và precision trong truy hồi.

| Hai tín hiệu | Ext-1, cùng ảnh → hai ảnh | Ext-2, cùng ảnh → hai ảnh |
|---|---:|---:|
| Điểm vùng tổng hợp và cosine CLIP gốc | 27,70 → 27,88 | 42,90 → 43,24 |
| Điểm qua từ của toàn ảnh và của vùng | 22,66 → 23,31 | 26,09 → 26,18 |

Ở hàng đầu, trong 2.321 lỗi top-1 trên Ext-1, việc tách ảnh sửa 7 trường hợp và làm hỏng 3. Trong 3.212 lỗi trên Ext-2, nó sửa 17 và làm hỏng 6. Ở hàng thứ hai, các cặp số tương ứng là 15/0 và 7/4. Khi giữ nguyên phân phối từng dấu hiệu, thay đổi cá thể cung cấp dấu hiệu hiếm khi đủ để đảo thứ tự hai lớp. Kết quả giới hạn lý do để xây một bộ ước lượng lớp ẩn phức tạp hơn cho chính phép gộp này.

Tệp JSON ghi kết quả trên từng nhóm và các đối chứng đơn. Bảng từng truy vấn được lưu theo hàm băm trong ZIP. Lượt đo trên Colab T4 dùng commit `235b4fe`. [Thiết kế chẩn đoán và nguồn nghiên cứu](../../CATEGORY_CONDITIONAL_EVIDENCE.md)
