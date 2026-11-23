# Kết quả kiểm tra các cặp từ qua hai prompt miền

Từ 20.931 tên công khai, phép tìm lân cận hai chiều xác định 3.699 cặp. Trung vị tương quan của log odds trên gallery giữa hai prompt là 0,913 ở Ext-1 và 0,911 ở Ext-2. Mức đồng thuận đã cao ở phần lớn các cặp. Trọng số theo tương quan thực cho 82,322 / 82,190 và 73,233 / 71,586 sau diffusion. Điểm gốc là 82,323 / 82,186 và 73,248 / 71,597. Xáo trọng số giữa các cặp vẫn cho 82,329 và 73,251 mAP. Không chọn biến thể để chạy TU.

Việc tráo hai từ gần nhau ở tất cả các cặp chỉ giảm 0,343 và 0,093 mAP sau diffusion. Cùng với chẩn đoán tráo tên gần nhau về tần suất ở nhánh GCWE, kết quả gợi ý cần phân biệt danh tính của nhóm khái niệm với danh tính chính xác của từng từ gần nghĩa. Hai thao tác đã tác động lên các tập từ và số lượng từ khác nhau. Một so sánh trực tiếp cần khớp độ phủ và tổng khối lượng xác suất bị hoán đổi trước khi kết luận hệ thống bất biến với từ đồng nghĩa.

[Thiết kế và prior art](protocol.md) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [Kết quả theo truy vấn](paired-L4.zip)
