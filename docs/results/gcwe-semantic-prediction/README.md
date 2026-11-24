# Có cần suy ra một từ để dự đoán vùng thứ ba của sketch?

Từ một góc A, mô hình đưa ra các giả thuyết về từ mô tả toàn cảnh. Từng từ chỉ định một phân bố trên các ảnh gallery có thể giải thích sketch. Sau khi thấy vùng B, mô hình cập nhật xác suất của từ rồi dự đoán phân bố từ tại vùng C chưa thấy. A, B và C là các crop không chồng pixel, được mã hóa riêng. Đối chứng giữ nguyên một ảnh để giải thích cả B và C có cùng hai phân bố dự đoán biên trước khi quan sát B.

| Phép so sánh, mô hình dùng từ trừ đối chứng | Ext-1 | Ext-2 |
|---|---:|---:|
| Log score vùng C so với bỏ qua quan sát B, nat | +0,01453 | +0,00816 |
| Log score vùng C so với giữ nguyên một ảnh cung cấp bằng chứng, nat | −0,05380 | −0,03666 |
| Log score vùng C so với dịch sai bố cục, nat | +0,00077 | +0,00063 |
| Xác suất từ của C nằm trong 64 giả thuyết ban đầu | 2,43% | 2,62% |

Cả 25 và 21 lớp đều có log score trung bình thấp hơn khi dùng từ chung thay cho một ảnh chung. Số liệu ủng hộ sự phụ thuộc giữa hai vùng. Cách giải thích bằng một từ trong 64 từ chọn từ góc A bỏ sót phần lớn khối lượng ngữ nghĩa. Cả hai mô hình được chấm bằng đầu ra CLIP của vùng C, chưa có chú thích bộ phận độc lập của con người.

Truy hồi trên 768 sketch mỗi split với toàn bộ gallery, dùng cùng diffusion và evaluator KD-SBIR. Hạng tử ngữ nghĩa đứng riêng sau cập nhật đạt **27,53 / 20,83** và **24,23 / 22,06** (mAP / precision). Trước cập nhật là **27,48 / 20,80** và **24,14 / 22,01**. Ngữ nghĩa toàn ảnh gốc trên chính mẫu đó đạt **82,79 / 82,46** và **75,66 / 74,13**. Phép thử tiếp theo mở đủ 20.931 từ và giữ phần còn lại của sketch dưới dạng bối cảnh.

[Mô hình sinh và đối chứng](protocol.md) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [Dữ liệu theo truy vấn](paired-T4.zip) · [Thí nghiệm mở rộng bối cảnh](../gcwe-semantic-reveal/README.md)
