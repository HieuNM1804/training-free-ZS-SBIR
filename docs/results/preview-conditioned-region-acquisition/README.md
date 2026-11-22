# Ảnh đọc ở độ phân giải gốc có dự đoán được kết quả phóng to?

Bộ mã hóa CLIP đọc sketch một lần. Ở khối cuối, ta giới hạn vùng mà token CLS được nhìn để tạo 13 quan sát độ phân giải gốc. Từ gallery không nhãn, phương pháp ước lượng những từ nào sẽ xuất hiện sau khi phóng to mỗi vùng dựa trên các ảnh có quan sát gốc tương tự. Khối cuối dùng toàn bộ patch tái tạo embedding CLIP gốc với sai khác cosine tối đa khoảng 0,000011.

| mAP / precision (%) trên mẫu | Sketchy Ext-1 | Sketchy Ext-2 |
|---|---:|---:|
| Một vùng chọn từ gallery với cùng số ảnh tham chiếu | 83,22 / 82,89 | 77,43 / 74,93 |
| Chọn theo quan sát gốc đúng vị trí, rồi đọc vùng phóng to | **83,82 / 83,75** | 78,15 / 75,45 |
| Hoán vị cặp gốc và phóng to giữa các ảnh cùng nhóm | 82,86 / 82,52 | **78,38 / 75,91** |
| Dùng quan sát của một vị trí khác trên sketch | 83,13 / 82,96 | **78,43 / 75,91** |
| Chỉ dùng quan sát gốc ở vị trí phương pháp chọn | 83,38 / 83,03 | 78,79 / 76,61 |
| Gộp đủ 13 vùng phóng to | 84,23 / 83,90 | 80,27 / 77,67 |

Cấu hình chọn để so retrieval trên mẫu dùng hệ số 20. Việc giữ đúng cặp gốc và phóng to trong mỗi ảnh cải thiện dự báo từ trên cả hai split. Ở hệ số 8, cross-entropy dự báo giảm từ 4,781 xuống 4,715 và từ 4,712 xuống 4,663 nat. Dự báo từ tốt hơn chưa đủ để chọn đúng vùng làm tăng mAP. Trên Ext-2 cả hai đối chứng vị trí sai và ghép sai ảnh đều vượt cấu hình chính.

Đây là 900 và 756 sketch mẫu, trên toàn bộ hai gallery. Mỗi nửa gallery giữ tối đa 96 ảnh tham chiếu cho từng nhóm, dùng cùng các ảnh cho các đối chứng ở trên. Cấu hình được thăm dò trên unseen. Nhánh dừng ở pilot, chưa có số đầy đủ trên TU-Berlin. [Cơ chế và tài liệu](../../PREVIEW_CONDITIONED_ACQUISITION.md) · [Kết quả các cấu hình](pilot.json) · [Dữ liệu từng truy vấn](pilot-per-query.zip)
