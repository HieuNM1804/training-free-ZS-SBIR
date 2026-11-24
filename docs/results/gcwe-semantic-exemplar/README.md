# Một ảnh mẫu chung có giúp hình học suy ra nghĩa tốt hơn?

Với từng patch làm tâm, CLIP cho biết patch gần tâm hay patch xa tâm giống nó hơn. Mỗi khái niệm công khai xác định một phân bố trên các ảnh tham chiếu. Thí nghiệm so hai giả thuyết ghép cặp: các quan sát của sketch được giải thích bởi cùng một ảnh mẫu ẩn, hoặc từng quan sát được giải thích bởi một ảnh khác thuộc cùng khái niệm. Khi chỉ có một quan sát, hai giả thuyết cho đúng cùng hậu nghiệm về mặt lý thuyết. Sự khác biệt khi tăng số quan sát đo đóng góp của việc giữ chung ảnh mẫu.

| Cấu hình, mAP / precision (%) | Ext-1 | Ext-2 |
|---|---:|---:|
| Khớp từ trực tiếp | 75,33 / 78,58 | 71,28 / 70,33 |
| Cùng ảnh mẫu, 16 quan sát | 75,34 / 78,54 | 71,32 / 70,36 |
| Độc lập, 16 quan sát | 75,34 / 78,54 | 71,32 / 70,36 |
| Cùng ảnh mẫu, 64 quan sát | 75,22 / 78,35 | 71,38 / 70,39 |
| Độc lập, 64 quan sát | 75,28 / 78,44 | 71,50 / 70,52 |

Ở 16 quan sát, cỡ mẫu hiệu dụng vẫn bằng khoảng 97,9% / 97,7% số ảnh tham chiếu. Bố cục mà sketch cung cấp chỉ thu hẹp rất ít tập ảnh có thể giải thích nó. Khi tăng lên 64 quan sát, cỡ mẫu hiệu dụng giảm về khoảng 76% ở cả hai split, nhưng mAP của cách dùng chung ảnh mẫu giảm 0,058 / 0,120 điểm so với lấy ảnh độc lập. Không có cấu hình không suy biến nào vượt đối chứng độc lập trên cả hai split. Dừng biến thể này ở pilot.

Mỗi split dùng 1.536 truy vấn, nguyên gallery, CLIP ViT-B/32 đóng băng, seed 42 và evaluator KD-SBIR trên Colab T4. [Nguyên lý, các công trình liên quan và tiêu chí chọn](protocol.md) · [Ext-1](sketchy_1-pilot.json) · [Ext-2](sketchy_2-pilot.json) · [Kết quả từng truy vấn](paired-T4.zip)
