# Giới hạn cập nhật xác suất theo hai cách gọi miền ảnh

Hai từ “drawing” và “sketch” tạo ra hai phân phối xác suất khác nhau cho cùng ảnh. Với mỗi tên trong từ điển, lấy khoảng chứa hai xác suất. Phân phối đề xuất là phân phối nằm trong tất cả các khoảng đó, có tổng bằng một và gần phân phối CLIP ban đầu nhất về KL. Có thể giải bằng một hệ số chung và phép chặn từng xác suất vào khoảng của nó. Chúng tôi cũng thử đặt cùng ràng buộc trực tiếp trên ảnh gallery. [Dẫn xuất, nguồn và đối chứng](../../DEPICTION_POSTERIOR_BOUNDS.md)

| mAP / precision (%) trên mẫu 36 sketch mỗi lớp | Sketchy Ext-1 | Sketchy Ext-2 |
|---|---:|---:|
| Mốc khớp từ và vùng | 84,64 / 83,96 | 80,63 / 78,38 |
| Giới hạn từng tên, cấu hình chính | 84,41 / 83,65 | 81,10 / 78,79 |
| Một cách gọi “drawing”, được chọn hệ số riêng | 84,42 / 83,67 | 81,10 / 78,88 |
| Co ngắn bước cập nhật tới cùng độ lệch KL | 84,44 / 83,68 | 81,06 / 78,74 |
| Một hệ số trộn hai cách gọi, được chọn riêng | 84,38 / 83,60 | 81,16 / 78,91 |

Cấu hình chính dùng hệ số 0,5 và điều chỉnh các vùng sketch trong không gian từ. Mức tăng Ext-2 xuất hiện cả với một prompt và bước cập nhật ngắn. Thí nghiệm chưa chỉ ra lợi ích riêng đủ lớn để giữ ràng buộc cho từng từ. Mỗi cấu hình cuối đã qua hàm đánh giá KD-SBIR gốc với 900 và 756 sketch cùng gallery đầy đủ. Nhánh dừng ở pilot Sketchy. [Kết quả và so sánh ghép cặp](pilot.json) · [Số liệu từng truy vấn](pilot-per-query.zip)
