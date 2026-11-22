# Chọn vùng bằng một nhóm ảnh và kiểm tra vùng bằng nhóm khác

Giữ nguyên từng sketch, chia gallery không nhãn thành sáu nhóm. Mỗi nhóm tự tìm các từ phổ biến và ảnh đại diện cho chúng. Một nhóm chọn vùng của sketch mà ảnh ứng viên có vẻ phù hợp nhất. Nhóm khác chấm các vùng ấy bằng chỉ mục của mình. Phương pháp cộng vào mốc mạnh ba lần chênh lệch log giữa cách đọc khác nhóm và cùng nhóm. Cấu hình dùng 32 từ mỗi nhóm, được chọn trên hai split Sketchy unseen và khóa trước TU-Berlin. [Công thức và công trình tham khảo](../../INDEPENDENT_GALLERY_REGION_READOUT.md) · [Cấu hình khóa](freeze.json)

| mAP / precision (%) | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin 30 lớp |
|---|---:|---:|---:|
| Mốc mạnh dùng từ và vùng | 84,37 / 83,85 | 77,02 / 74,83 | 80,07 / 80,92 |
| Tách nhóm và cộng tỷ số vào điểm gốc | 84,33 / 84,20 | 77,59 / 75,28 | 79,53 / 80,30 |
| Chọn và chấm vùng bằng cùng nhóm, lấy trung bình, chọn hệ số riêng | 84,37 / 83,80 | 77,01 / 74,81 | 80,16 / 80,99 |

Ở Ext-2, phương pháp sửa 122 truy vấn sai hạng nhất và làm sai 47 truy vấn. Mức tăng +0,57 điểm mAP có khoảng bootstrap 95% theo lớp [−0,05, +1,33]. Trên TU-Berlin, phương pháp sửa 18 truy vấn và làm sai 56, mAP giảm 0,55 điểm với khoảng [−1,15, −0,06]. Trên hai gallery có một phần ảnh trùng, cùng một cặp đúng–sai đổi thứ tự 1,35% sau can thiệp, so với 0,60% ở mốc. Thử lại với hai cách chia ngẫu nhiên khác cũng làm mất phần lớn mức tăng quan sát ở pilot Ext-2. Các kết quả này bác bỏ cách giải thích rằng việc tách nhóm ổn định hóa thứ hạng đủ để cải thiện chuyển bộ. [Kiểm chứng cơ chế và độ nhạy với lần chia](REPLICATION_VI.md)

Đã chấm tất cả 14.179, 12.694 và 2.130 sketch bằng hàm gốc KD-SBIR, trên gallery 17.100, 12.552 và 27.262 ảnh tương ứng. Ext-1 và TU-Berlin dùng mAP toàn gallery và P@100. Ext-2 dùng mAP@200 và P@200. Ext-1 chạy trên L4. Sau khi Colab đổi phiên, Ext-2 và TU-Berlin chạy trên T4 từ cùng cấu hình khóa. Nhánh dùng 20.931 tên Open Images và OpenAI CLIP ViT-B/32 đóng băng. [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU-Berlin](tuberlin.json) · [Tất cả điểm từng sketch](per-query.zip)
