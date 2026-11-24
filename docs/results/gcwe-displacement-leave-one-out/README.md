# Ablation trên cấu hình GCWE theo độ dịch chuyển, bỏ Hierarchical

Tất cả các hàng dùng cùng CLIP ViT-B/32 đóng băng và seed 42 trên Google Colab T4. Cả ba bộ đều chấm toàn bộ truy vấn với evaluator KD-SBIR. Có 14.179 sketch / 17.100 ảnh ở Ext-1, 12.694 / 12.552 ở Ext-2 và 2.130 / 27.262 ở TU-Berlin 30 lớp. Hai split Sketchy đã tham gia phát triển. Tất cả hệ số giữ nguyên khi bỏ từng thành phần và khi chuyển sang TU.

| Cấu hình | Ext-1 mAP@all / P@100 | Ext-2 mAP@200 / P@200 | TU mAP@all / P@100 |
|---|---:|---:|---:|
| Đủ ba thành phần | **85,086 / 84,600** | **78,406 / 76,170** | **81,870 / 82,981** |
| Bỏ hạng tử ngữ nghĩa toàn ảnh | 82,429 / 81,807 | 75,125 / 73,150 | 75,715 / 75,460 |
| Bỏ hạng tử vùng theo ứng viên | 82,994 / 82,346 | 76,201 / 74,416 | 81,583 / 82,905 |
| Bỏ hạng tử GCWE theo độ dịch chuyển | 84,627 / 84,082 | 77,555 / 75,328 | 80,569 / 81,466 |

Điểm đầy đủ là 0,65S + 0,35R + 0,65G. S là log điểm từ của toàn ảnh sau diffusion. R là log điểm vùng theo ứng viên. G là thay đổi log điểm từ do GCWE cập nhật phân bố từ từ bằng chứng dịch chuyển. Bỏ S vẫn cho phép các hạng tử còn lại dùng từ điển chung. Đối chứng GCWE gốc dùng để so khớp số liệu cũng được lưu trong JSON. [Mã scoring](../../../tf_sbir/gcwe_displacement_leave_one_out.py) · [Protocol](protocol.md)

| Hạng tử bỏ đi | Giảm mAP Ext-1 | Giảm mAP Ext-2 | Giảm mAP TU | Khoảng 95% theo lớp của hiệu ứng trên TU |
|---|---:|---:|---:|---:|
| S | 2,656 | 3,281 | 6,155 | [3,544, 9,144] |
| R | 2,091 | 2,206 | 0,286 | [−0,679, 1,203] |
| G | 0,458 | 0,851 | 1,300 | [0,519, 2,169] |

Khoảng bootstrap cho trọng số bằng nhau theo lớp, còn metric chính thức trung bình theo truy vấn. Trên TU cả 30 lớp đều được chấm. S và G còn thể hiện lợi ích ở split chuyển giao, trong khi mức giảm khi bỏ R nhỏ hơn hai split phát triển. Ablation ước lượng đóng góp của hạng tử trong công thức này. Nó chưa xác định được nguyên nhân của khác biệt giữa các bộ dữ liệu. [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [TU](tuberlin.json)

**Kiểm tra tái lập.** Cố định Python, NumPy, PyTorch, CUDA, seed 42 cho phép chia gallery và bootstrap, bật thuật toán deterministic và cấu hình workspace cuBLAS. Scorer đối chiếu trực tiếp AP và precision với hàm KD gốc ở từng batch. Hai split Sketchy khớp ngưỡng chặt khi so với lần chấm cấu hình mạnh trước đây. Trên TU, so với lượt lưu cũ, hàng đầy đủ có 33 AP và 1 precision vượt ngưỡng gần bằng theo truy vấn; thay đổi trung bình là +0,000028 điểm mAP và +0,000469 điểm P@100. Bản tổng hợp đầu dừng ở kiểm tra này sau khi đã ghi đủ từng batch. Đã phục hồi toàn bộ hàng theo ID và hash cấu hình, giữ cả thông tin sai khác. Một tiến trình thứ hai chấm lại toàn bộ TU từ cùng cache và seed. Các precision của hàng đầy đủ giống hệt ở hai lượt mới. 169 AP có khác biệt số học, sai khác tuyệt đối trung bình là 0,000468 điểm. Hàng bỏ R có một sketch lệch 1 điểm P@100. Các số ở bảng trên lấy từ lượt thứ nhất. Thực nghiệm tái lập ở độ chính xác báo cáo thông thường; chưa xác định nguyên nhân của từng chênh lệch số học. [So sánh giữa hai tiến trình](tuberlin-repeat.json) · [Dữ liệu từng truy vấn, gồm cả lần lặp TU](paired-T4.zip)
