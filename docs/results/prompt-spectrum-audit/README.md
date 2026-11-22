# Điểm thay đổi do đổi thứ tự tên hay do thay đổi phân bố điểm?

Cùng can thiệp attention head đã khóa, ta có hai danh sách điểm CLIP cho 20.931 tên, trước và sau chỉnh sửa. Có thể giữ nguyên toàn bộ dãy điểm cũ rồi gán chúng cho các tên theo thứ tự mới. Cách này giữ nguyên entropy. Ngược lại, có thể giữ thứ tự tên cũ và thay bằng dãy giá trị mới. Hai phép thử xác định phần nào của biểu diễn đã tác động đến ảnh truy hồi. [Định nghĩa các can thiệp](../../PROMPT_SPECTRUM_AUDIT.md) · [Cấu hình gốc](../paired-prompt-head-intervention/FULL_VI.md)

| mAP / precision (%) | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin 30 lớp |
|---|---:|---:|---:|
| Mốc vùng chưa chỉnh sửa | 84,37 / 83,85 | 77,01 / 74,84 | 80,07 / 80,92 |
| Chỉ đổi thứ tự gán tên | 84,81 / 84,34 | 77,12 / 75,00 | 80,00 / 80,48 |
| Chỉ đổi dãy giá trị điểm | 84,38 / 83,91 | 77,09 / 74,89 | 79,75 / 80,52 |
| Can thiệp đầy đủ đã khóa | 84,83 / 84,41 | 77,27 / 75,09 | 79,65 / 80,16 |

Ở Ext-1, riêng việc đổi thứ tự tên đã tăng 0,44 điểm mAP trong tổng mức tăng 0,47. Ở TU-Berlin, riêng việc đổi dãy điểm làm giảm 0,33 điểm mAP trong tổng mức giảm 0,43. Hai thành phần có tương tác qua xếp hạng ảnh nên mức tác động riêng có thể không cộng đúng bằng thay đổi chung. Trong các truy vấn vốn sai ở vị trí đầu có tên lớp khớp một từ công khai, Ext-1 có 1.466 trường hợp tên lớp tăng hạng và 598 trường hợp giảm hạng. Ext-2 có 1.263 và 1.458 trường hợp, TU-Berlin có 119 và 177 trường hợp tương ứng. Tên lớp chỉ xuất hiện trong chẩn đoán hậu nghiệm.

Trên một sketch chiếc cưa ở Ext-2, thứ hạng đầu đổi từ kiếm sang cưa, AP@200 tăng từ 1,38 lên 19,53 dù từ đứng đầu vẫn là “ball-peen hammer”. Tên “saw” đi từ hạng 1.169 lên 996. Ở sketch tê giác, ảnh đầu đổi từ tê giác sang bò và AP@200 giảm từ 100,00 xuống 59,29. Hai ví dụ được chọn tự động theo trung vị thay đổi AP trong các nhóm sửa đúng và làm sai hạng đầu, không dùng để chọn cấu hình. [Chi tiết Ext-2](sketchy_2.json)

Đây là phân tích hậu nghiệm của một cấu hình đã biết kết quả trên cả ba benchmark. Mọi sketch và mọi điều kiện đều được chấm bằng hàm KD-SBIR gốc. Hai điều kiện đầu cuối tái lập đúng từng giá trị mAP và precision theo truy vấn của lần chạy đã lưu. [Ext-1](sketchy_1.json) · [TU-Berlin](tuberlin.json) · [Điểm từng sketch](per-query.zip)
