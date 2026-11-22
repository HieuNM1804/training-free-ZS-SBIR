# Cơ chế ở tầng đầu và kết quả truy hồi

Trên Colab T4, với lượng chi tiết tự nhiên khôi phục từ phép đo patch, thành phần chung cho hai dấu chiếm phần lớn thay đổi embedding. Phát lại phần còn lại của CLIP từ điểm giữa hai trạng thái cho thấy hiệu ứng xuất hiện sớm. Sau block 2, tỷ số chiếu trung vị lên phản ứng cuối là 0,86 cho 240 sketch và 0,70 cho 240 ảnh. Sau block 4, cả hai tỷ số xấp xỉ 1,00. Tách nhỏ hơn cho thấy riêng MLP của block 2 tạo phần chiếu trung vị 0,86 trên toàn bộ 1.644 sketch pilot duy nhất và 0,73 trên 240 ảnh. Với kích thích đẳng hướng bất kỳ, tương quan theo tầng đổi khác. Các giá trị là chiếu có dấu lên một vector tham chiếu, có thể vượt 1. Chúng không biểu thị độ chính xác.

Trong một lượt kiểm tra độc lập, 240 ảnh gallery ngẫu nhiên dùng để định danh neuron, 240 ảnh gallery khác và 240 sketch để đối chứng. Có 3.072 neuron ẩn mỗi MLP. Tám neuron được chọn ở block 2 chiếm trung vị 92,3% tổng đóng góp tuyệt đối vào chênh lệch cục bộ ở ảnh giữ lại và 92,2% ở sketch. Con số của tám neuron có hoạt hóa tự nhiên lớn nhất là khoảng 40% và 47%. Ở block 3, tám neuron được chọn chiếm 91,4% và 84,5%. Khi dùng nhiễu đẳng hướng trực tiếp trên token, tám neuron chọn theo chính kích thích đó chỉ chiếm khoảng 21,6% ở ảnh và 25,0% ở sketch tại block 2. Các con số tập trung mô tả chênh lệch ở MLP sau một loại kích thích cụ thể.

Điểm % dưới đây theo mAP / precision, dùng evaluator KD-SBIR gốc. Hai pilot Sketchy có 900 và 756 truy vấn, toàn gallery. Những cấu hình này được chọn trực tiếp trên hai split unseen.

| Phương án | Ext-1 | Ext-2 |
|---|---:|---:|
| Nền vùng và từ công khai | 84,64 / 83,94 | 80,64 / 78,38 |
| Phần tăng thêm trong attention của block 3, hai hướng đẳng hướng | 84,66 / 83,74 | 81,01 / 78,65 |
| Trung bình đầu ra hai dấu, tự chọn hệ số | 84,68 / 84,23 | 80,39 / 78,27 |
| Dùng một dấu, tự chọn hệ số | 84,69 / 84,27 | 80,36 / 78,26 |
| Toàn bộ hiệu ứng đến hết block 4 | 84,77 / 84,56 | 80,89 / 78,70 |
| Phần tăng thêm ở block 2, kích thích từ hệ số thật | 84,77 / 84,24 | 80,80 / 78,49 |
| Cùng phép đo block 2, hệ số bị tráo | 84,79 / 84,25 | 80,79 / 78,52 |

Phương án đứng đầu theo trung bình mAP sử dụng phần thay đổi bắt đầu ở attention block 3, qua từ công khai, hệ số 1,5. Nó sửa/hỏng 5/7 top-1 ở Ext-1 và 10/12 ở Ext-2. Chênh mAP với trung bình hai đầu ra là −0,02 điểm ở Ext-1 và +0,62 điểm ở Ext-2. Khoảng tin cậy 95% theo lớp đều bao gồm 0. Kết quả cho thấy một nhóm neuron mang đáp ứng lặp lại ở cả hai miền. Điểm truy hồi chưa chứng minh sửa trực tiếp đáp ứng đó tạo được cải thiện ổn định. Chưa chấm TU-Berlin cho can thiệp này.

JSON và ZIP đi kèm lưu kết quả từng truy vấn, từng neuron và đối chứng phát lại. [Công thức và nguồn nghiên cứu](../../PATCH_RESPONSE_MEDIATION.md)
