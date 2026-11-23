# Baseline ba thành phần, Colab L4, seed 42

Cấu hình ghép GCWE, Candidate-Conditioned Regional Matching và Cross-Domain Semantic Retrieval. Nền tảng chung là OpenAI CLIP ViT-B/32 đóng băng, từ điển 20.931 từ Open Images và suy luận độc lập cho từng sketch. Dữ liệu, hai hướng nhìn, gallery và hàm tính metric giữ nguyên trong từng phép bỏ. Mỗi lượt Colab chạy từ một tiến trình Python mới. Sketchy unseen từng được dùng để phát triển các hệ số; TU được dùng để kiểm tra chuyển giao.

| Cấu hình | Ext-1 mAP@all / P@100 | Ext-2 mAP@200 / P@200 | TU mAP@all / P@100 |
|---|---:|---:|---:|
| Đủ ba thành phần | 84,78 / 84,24 | 77,84 / 75,66 | 81,49 / 82,20 |
| Bỏ cập nhật từ bằng hình học | 84,63 / 84,08 | 77,55 / 75,33 | 80,57 / 81,47 |
| Bỏ điểm khớp vùng | 82,73 / 82,16 | 75,37 / 73,89 | 80,70 / 81,23 |
| Bỏ diffusion RGB trong cả hai luồng dùng nó | 80,29 / 83,61 | 76,10 / 74,44 | 77,78 / 80,48 |

Ở dòng cuối, điểm khớp từ toàn ảnh vẫn hoạt động. Hai nhánh còn lại cũng cần từ điển để tính điểm nội tại. Đối chứng tắt hạng tử ngữ nghĩa toàn ảnh là một câu hỏi riêng. Số sketch/ảnh lần lượt là 14.179/17.100, 12.694/12.552 và 2.130/27.262, gồm đủ 30 lớp TU.

Trên hai lần chấm lại với cùng cache đặc trưng, sai lệch tuyệt đối lớn nhất của các metric trung bình là 0,0000191 điểm phần trăm. Riêng dòng đầy đủ, có 35, 0 và 18 truy vấn với ít nhất một metric khác ở độ chính xác lưu trữ. Dòng bỏ diffusion khớp từng truy vấn trên cả ba split. Điều này thu hẹp phạm vi kiểm tra nguồn sai số, chưa chứng minh nguyên nhân nằm ở một kernel GPU cụ thể. [Dữ liệu truy vấn hai lần chạy](seed42-L4.zip) · [Sai lệch](repeat-differences.json)

Bảng 1 của SpLIP báo cáo lần lượt 80,2 / 86,7, 76,4 / 77,3 và 73,1 / 78,2 với cùng tên split và cutoff. Baseline hiện cao hơn 4,58 / 1,44 / 8,39 điểm mAP, nhưng precision thấp hơn 2,46 và 1,64 điểm ở hai split Sketchy. Đây là đối chiếu với số tác giả công bố. SpLIP có học prompt, pipeline hiện dùng từ điển ngoài và thống kê gallery, còn các tham số lịch sử được chọn trên Sketchy unseen. Mức chênh này chưa đủ để kết luận xếp hạng SOTA. [SpLIP, ECCV 2024, bảng 1](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03426.pdf)
