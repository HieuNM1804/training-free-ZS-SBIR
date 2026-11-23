# So sánh bằng chứng toàn cảnh và từng phần qua từ công khai

Chấm trên Colab T4, dùng 36 truy vấn mỗi lớp và toàn gallery của từng split, tổng cộng 900 và 756 truy vấn. Các hệ số được chọn trên cùng hai pilot unseen. Mỗi ô là mAP / precision theo phần trăm, Ext-1 dùng mAP@all / P@100, Ext-2 dùng mAP@200 / P@200.

| Cấu hình | Sketchy Ext-1 | Sketchy Ext-2 |
|---|---:|---:|
| Nền vùng và từ công khai | 84,636 / 83,944 | 80,644 / 78,375 |
| Tương tác hai phía toàn ảnh × từng phần | 84,792 / 83,950 | 80,638 / 78,439 |
| Tương phản chỉ ở phía sketch | 85,083 / 84,146 | 80,480 / 78,396 |
| Phạt điểm khớp giữa các mảnh | 85,178 / 84,751 | 80,954 / 78,269 |

Độ tăng mAP của tương tác hai phía là +0,156 và −0,006 điểm phần trăm. Đối chứng phạt trực tiếp các điểm khớp giữa mảnh cho mAP cao hơn ở cả hai split. Pilot ủng hộ kiểm tra riêng tín hiệu nuisance của từng truy vấn. Phép phát lại thứ hạng trên các cấu hình được chọn có sai số mAP/precision tối đa 2,39 × 10⁻⁷ so với hàm KD-SBIR gốc. File `pilot.json` lưu danh sách cấu hình đã phát lại, khoảng tin cậy theo lớp và sai số tính lại từng split. File nén chứa AP, precision và chẩn đoán theo truy vấn cho toàn lưới pilot. Chưa đánh giá nhánh này trên toàn bộ Sketchy hay TU-Berlin.
