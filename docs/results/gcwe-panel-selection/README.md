# Bằng chứng dùng khi chọn ảnh đã giải thích được bao nhiêu?

Trên đúng 768 sketch mỗi split của thử nghiệm ghép quan hệ, thay nguồn quyết định danh sách ảnh trước khi suy từ. Số ảnh duy nhất trong mỗi panel được giữ nguyên từng trường hợp. Vùng thứ tư dùng để chọn ảnh có pixel tách biệt với hai vùng quan sát và vùng chấm điểm. Đối chứng mù sử dụng ảnh lấy đồng đều với RNG đã khóa theo truy vấn và nửa gallery.

| Nguồn chọn ảnh | Lợi ích của suy luận so với phân bố trước cập nhật, Ext-1 | Ext-2 |
|---|---:|---:|
| Chính hai vùng dùng ở bước cập nhật | −0,00793 | −0,00634 |
| Vùng thứ tư của sketch | +0,00203 | −0,00292 |
| Chọn ảnh ngẫu nhiên | +0,00017 | +0,01533 |

Đơn vị là nat của log score trên vùng C chưa thấy. Chuyển sang vùng thứ tư làm lợi ích biên thay đổi +0,00996 / +0,00342 nat. Tuy vậy mức tăng tuyệt đối so với phân bố ban đầu chỉ dương trên một split. Với panel ngẫu nhiên, ghép hai ảnh vượt cách dùng một ảnh, nhưng hoán vị bỏ cấu trúc lân cận lại cho dự báo tốt hơn trên cả hai split. Thay đổi panel tác động cả nội dung ảnh lẫn việc dùng lại bằng chứng. Do đó chưa thể quy kết thất bại ban đầu cho một nguyên nhân duy nhất hoặc nhận đây là cải tiến GCWE. [Protocol](protocol.md) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [Các đối chứng trên cùng truy vấn](paired-T4.zip)
