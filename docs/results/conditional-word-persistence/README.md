# Có cần theo dõi danh tính của từ suốt các quyết định trên cây gallery?

Một sketch cho phân phối tên công khai. Mỗi lần đi từ một nhóm ảnh đến nhóm con, từ đang giải thích sketch được giữ lại, hoặc được thay bằng một tên mới. Mô hình đề xuất tính xác suất này riêng cho từng từ từ hai phần gallery độc lập. Một đối chứng giữ đúng kỳ vọng tổng khối lượng được giữ ở mỗi nút, rồi phân bố đều xác suất giữ đó cho các từ. Cả hai dùng cùng cây, bộ từ, phân bố ban đầu và gallery. [Phương trình, nguồn nghiên cứu và điều kiện bác bỏ](../../CONDITIONAL_WORD_PERSISTENCE.md)

Chấm trên Colab L4 với OpenAI CLIP ViT-B/32 đóng băng và một sketch mỗi lượt. Các hệ số được chọn bằng mAP trung bình của hai split unseen Sketchy, khóa trước khi chạy TU-Berlin. Mỗi ô là mAP / precision theo %. Ext-1 và TU dùng mAP@all / P@100, Ext-2 dùng mAP@200 / P@200. Toàn bộ 14.179 / 12.694 / 2.130 truy vấn được chấm trên 17.100 / 12.552 / 27.262 ảnh, gồm đủ 30 lớp TU. [Commit khóa cấu hình](https://github.com/HieuNM1804/training-free-ZS-SBIR/commit/6a889a811efc3db762cd8bd3c789cd6982df83c9)

| Phương pháp | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin |
|---|---:|---:|---:|
| Giữ một từ suốt đường đi, chọn cấu hình riêng | 80,43 / 81,42 | 74,35 / 71,89 | 77,36 / 79,55 |
| Chọn lại từ ở mỗi quyết định, chọn cấu hình riêng | 80,57 / 81,37 | 74,32 / 72,06 | 77,26 / 79,34 |
| **Xác suất giữ theo từng từ, từ mới theo khả năng phân biệt** | **80,56 / 81,47** | **73,61 / 71,68** | **76,90 / 78,89** |
| Khớp tổng khối lượng giữ, bỏ danh tính của từ, cùng tham số | 80,59 / 81,53 | 73,72 / 71,76 | 76,99 / 78,97 |
| Nền vùng/từ và diffusion | 84,37 / 83,85 | 77,01 / 74,84 | 80,07 / 80,92 |
| Thêm phần đổi trạng thái từ tốt nhất trên Sketchy vào nền | 84,36 / 83,87 | 77,26 / 74,93 | 80,19 / 81,07 |

Mô hình theo từng từ thấp hơn đối chứng khớp khối lượng 0,029 / 0,113 / 0,094 điểm mAP. Cả ba khoảng bootstrap 95% theo lớp đều chứa 0. Cấu hình bổ sung trên nền vùng/từ chọn hệ số −0,5, tức đảo hướng điều chỉnh do việc đổi trạng thái sinh ra. Kết quả chưa chứng minh việc xác định danh tính từ nào phải giữ lại là đóng góp làm tăng mAP. Hạn chế của cấu hình chính là dùng 32 từ đứng đầu của một truy vấn, còn các đối chứng được phép chọn 128 từ. Hàng khớp khối lượng dùng cùng đúng 32 từ.

Đồng nhất thức khi giữ một từ khôi phục điểm khớp từ trực tiếp ở cấp nhóm với sai số log tuyệt đối tối đa 3,89 × 10⁻⁵ và sai số bảo toàn khối lượng dưới 7,5 × 10⁻⁷. Dùng một từ duy nhất khiến mọi cơ chế lấy lại từ cho cùng một kết quả. 42 thứ hạng cuối được phát lại trên evaluator KD-SBIR gốc với sai số tối đa 2,39 × 10⁻⁷ so với phép tính vector hóa. JSON và ZIP đi kèm lưu kết quả và hai metric theo từng truy vấn.
