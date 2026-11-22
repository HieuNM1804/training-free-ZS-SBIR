# Đổi thứ hạng có làm mất ảnh của một cách hiểu còn khả dĩ?

Với mỗi sketch, mô hình xem thứ tự xuất hiện của ảnh thuộc từng nhóm thị giác không nhãn. Nó ước tính chi phí khi các ảnh của một nhóm bị đẩy sâu xuống danh sách. Khi nhiều ảnh có thể thay thế nhau, lần xuất hiện thứ j của một nhóm được so với lần xuất hiện thứ j của cùng nhóm trong thứ hạng gốc. Chi phí này được tính theo cả bằng chứng từ vựng và cosine thị giác của sketch. [Giao thức, công thức và tiền lệ](../../SEMANTIC_COVERAGE_TRANSPORT.md) · [Lỗi dẫn tới phép thử](https://github.com/HieuNM1804/training-free-ZS-SBIR/blob/03fd77a/docs/results/latent-intent-ranking/DECISION_DIAGNOSIS_VI.md)

| mAP / precision (%) | Sketchy Ext-1 | Sketchy Ext-2 |
|---|---:|---:|
| Điểm nền theo vùng | 84,64 / 83,96 | 80,63 / 78,38 |
| Chi phí mất độ phủ theo nhóm ảnh | 84,67 / 83,99 | 80,62 / 78,33 |
| Tính theo từng ảnh riêng biệt, cùng tham số | 84,64 / 83,96 | 80,69 / 78,40 |
| Bỏ chi phí thiệt hại, giữ cùng tham số còn lại | 81,57 / 82,93 | 75,33 / 73,34 |
| Bỏ chi phí thiệt hại và dò lại tham số | 84,33 / 84,06 | 79,48 / 77,20 |
| Mức pha cố định, dò riêng trên hai mẫu | 84,67 / 83,97 | 80,67 / 78,38 |
| Oracle chọn theo AP thực tế của từng sketch | 87,99 / 89,84 | 90,76 / 86,66 |

Cấu hình chính chỉ đổi 9 thứ hạng trên 900 sketch của Ext-1 và 3 trên 756 sketch của Ext-2. Một trong ba sketch đã đổi ở Ext-2 mất 45,89 điểm AP. Hệ thống bảo toàn phần lớn thứ hạng bằng cách rất ít khi can thiệp. Cùng 51 phương án xếp hạng, oracle hậu nghiệm cho thấy còn nhiều cơ hội cải thiện ở Ext-2. Phép đo từ các nhóm ảnh hiện tại chưa phân biệt được những cơ hội này. Các kết quả là pilot cân bằng 36 sketch mỗi lớp. Phương pháp dừng ở Sketchy.

Mỗi phương án cho mỗi sketch đều được chấm bằng hàm KD-SBIR gốc trên Colab L4, với 17.100 ảnh ở Ext-1 và 12.552 ảnh ở Ext-2. Ext-1 dùng mAP@all và P@100, Ext-2 dùng mAP@200 và P@200. [Các cấu hình và điểm ghép cặp](pilot.json) · [Ext-1](pilot-sketchy_1.json) · [Ext-2](pilot-sketchy_2.json) · [Thứ hạng được mã hóa qua điểm của từng hành động và từng sketch](pilot-per-query.zip)
