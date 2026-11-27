# Sửa vector sketch có tái hiện lợi ích của 64 láng giềng ảnh không?

Trên cả ba tập, cho phép dùng gallery đích không nhãn để xác định phép sửa vector sketch tái hiện được 89–90% phần tăng mAP của 64-NN. Khi cố định phép sửa từ Ext-1, điểm giảm trên cả Ext-2 và TU-Berlin. Hai kết quả trả lời hai câu hỏi khác nhau về khả năng biểu diễn và khả năng chuyển giao.

## Cách đo

Mỗi sketch cho một vector `q` trên 601 từ Open Images boxable. CLIP ViT-B/32 đọc sketch và bản lật ngang, áp softmax trên từ cho từng lượt rồi lấy trung bình. Mỗi ảnh gallery cho một hàng trong ma trận `P`. Template là `a drawing of a {}.` và `a photo of the {}.`. Semantic Dot cho điểm `Pq`. Với từng ảnh, lấy trung bình vector từ của 64 ảnh **khác** gần nhất theo đặc trưng RGB CLIP 512 chiều, được ma trận `B = WP`. Các điểm `Bq` cũng bằng `W(Pq)` cho mọi sketch.

Để thử sửa riêng `q`, giữ nguyên `P` và tìm một phép biến đổi từ gallery sao cho điểm mới gần `Bq` nhất theo tổng bình phương sai số. Dùng `X = [1, P[:, :-1]]` vì mỗi hàng của `P` cộng thành 1. Chiếu `B` trực giao lên các cột của `X` được `F = XA`. Có thể tính `Fq` chính xác bằng cách đổi vector sketch rồi lấy dot với `P`. Tọa độ sketch sau phép đổi có thể âm. Phép chiếu tối ưu cho sai số điểm. Thứ tự xếp hạng là một đại lượng khác. Nghiệm không sử dụng nhãn hoặc các sketch khác.

Kết quả chính dùng toàn bộ unseen trên Kaggle NVIDIA L4, seed 42. Số sketch/ảnh lần lượt là 14.179/17.100, 12.694/12.552 và 2.130/27.262. Mỗi ô là mAP / precision theo phần trăm. Ext-1 và TU dùng mAP@all / P@100. Ext-2 dùng mAP@200 / P@200. Mã kiểm tra hash hàm KD-SBIR gốc và đối chiếu từng hàng với kết quả của evaluator đó. Hai split Sketchy là thăm dò. `k=64` là cấu hình có từ đợt trước. Lượt này cố định phép đo trước khi đọc bảng chính và không dò tham số.

| Điều kiện | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin 30 lớp |
|---|---:|---:|---:|
| Semantic Dot | 75,35 / 77,10 | 72,66 / 71,07 | 73,46 / 73,98 |
| Trung bình 64 ảnh cùng gallery | 80,15 / 79,94 | 75,71 / 73,60 | 78,12 / 76,50 |
| Sửa sketch từ gallery đang chấm | 79,61 / 79,82 | 75,42 / 73,50 | 77,66 / 76,52 |
| Thêm riêng phần dư trực giao vào Dot | 75,03 / 77,58 | 73,19 / 71,58 | 72,97 / 74,45 |
| Khóa phép sửa từ Ext-1 | 79,61 / 79,82 | 59,83 / 56,04 | 44,62 / 40,75 |

Phép sửa trên cùng gallery tăng 4,26 / 2,75 / 4,20 điểm mAP so với Dot. Khoảng cách mAP còn lại tới 64-NN là 0,54 / 0,29 / 0,46 điểm. Phần dư ngoài khả năng của phép sửa `q` vẫn chiếm 37,97% / 30,64% / 40,48% tổng bình phương thay đổi điểm sau khi trừ trung bình theo ảnh. Vì vậy tỷ lệ 89–90% chỉ diễn tả phần tăng mAP. Chẩn đoán này sử dụng kết quả 64-NN làm đích tham chiếu trong cùng gallery và chưa đáp ứng mục tiêu về một phương pháp độc lập.

RGB diffusion là một đối chứng lịch sử khác. Với cấu hình từ và query của nhánh đó, điểm trực tiếp là 75,31 / 71,42 / 73,35 mAP. Dùng đồ thị RGB cho 81,86 / 74,20 / 79,16. Hai nhánh cùng cho thấy giá trị của quan hệ ảnh–ảnh nhưng phải đọc mức tăng so với đối chứng trực tiếp của từng nhánh. [Ablation RGB diffusion](https://github.com/HieuNM1804/training-free-ZS-SBIR/blob/3e0e4843c50bef434d9091b8cc96d5abf3ca49e9/docs/results/cross-diffusion-map/summary.json) · [Đối chứng 64 ảnh gốc](https://github.com/HieuNM1804/training-free-ZS-SBIR/blob/914da129f914eb8dfed45cffb8bb0dca4178856e/docs/results/conditional-sphere-completion/full-unseen.json)

## Cùng sketch, cùng ảnh đúng, chỉ thêm ảnh sai

Phân tích sau khi có bảng chính. Hai split Sketchy chia sẻ 2.231 sketch và 2.205 ảnh của bốn lớp scissors, tree, wheelchair và windmill, xác nhận bằng ID của từng ảnh. Với cùng các sketch này, xếp hạng 2.205 ảnh chung, rồi thêm toàn bộ 10.347 ảnh của 17 lớp còn lại ở Ext-2. Tất cả ảnh thêm vào đều sai đối với mọi truy vấn đang xét. Cả hai lần dùng cùng mAP@200 / P@200. Tất cả phép biến đổi trong mỗi dòng được giữ cố định khi mở rộng gallery xếp hạng.

| Điều kiện | Chỉ 2.205 ảnh chung | Đủ 12.552 ảnh Ext-2 | Số ảnh thêm lọt top 200, trung bình | Truy vấn có ảnh thêm đứng đầu |
|---|---:|---:|---:|---:|
| Semantic Dot | 86,88 / 86,34 | 74,38 / 72,77 | 41,95 | 20,62% |
| 64-NN của Ext-2 | 89,28 / 88,74 | 78,28 / 75,19 | 38,54 | 16,67% |
| Sửa sketch từ Ext-2 | 88,87 / 88,10 | 77,78 / 74,36 | 39,88 | 16,94% |
| Sửa sketch từ Ext-1 | 89,47 / 88,54 | 61,28 / 60,50 | 70,37 | 73,33% |

Mở rộng gallery làm mức chênh mAP của phép sửa từ Ext-1 so với Dot chuyển từ **+2,60** thành **−13,11** điểm cho chính các sketch đó. Chênh lệch tọa độ của cùng ảnh sau khi áp phép sửa nguồn dưới hai cách tính nhỏ hơn 2×10⁻¹⁴. Trên 10.463 sketch thuộc 17 lớp còn lại của Ext-2, Dot đạt 72,30, sửa từ Ext-1 đạt 59,53, sửa từ Ext-2 đạt 74,91 mAP. Hiện tượng vì vậy có cả ở những sketch và lớp đã xuất hiện khi so sánh hai split.

Với 2.205 ảnh chung, trung bình có 59,74/64 láng giềng trùng ID khi xây hai đồ thị độc lập, trung vị là 64. Khi thay gallery dùng để tạo trung bình nhưng giữ danh sách ứng viên là những ảnh chung, mAP@200 đổi từ 90,11 sang 89,28. Đây là phép so cho bốn lớp và một tập ứng viên rút gọn. Bảng full unseen ở trên mới là con số benchmark.

## Ảnh sai nào được đẩy lên?

Với vector từ của từng ảnh đích ở dạng hàng `x`, đo `h(x) = x (X_sᵀ X_s)⁻¹ xᵀ`, trong đó `X_s` chỉ chứa ảnh Ext-1. Số này lớn khi ảnh dùng các tổ hợp tọa độ từ ít được đại diện trong gallery nguồn. Có một lý do để kiểm tra nó. Giữ nguyên `X_s` và thay đổi điểm tham chiếu nguồn cho một sketch một lượng `δy`, bất đẳng thức Cauchy–Schwarz cho `|δs(x)| ≤ √h(x) ‖δy‖₂` đối với phép ngoại suy bình phương tối thiểu. Đây là cận độ nhạy. Sai số truy hồi thực tế phải đo trên ảnh và truy vấn.

Trung vị `h` là 0,0116 cho 2.205 ảnh chung và 0,6273 cho 10.347 ảnh thêm. Với **từng** sketch của bốn lớp chung, tìm ảnh đứng cao nhất trong 10.347 ảnh sai, rồi tính phần trăm ảnh trong **chính lớp của ảnh sai đó** có leverage thấp hơn nó, chia đôi các trường hợp bằng nhau. Mức trung bình là 33,5 ở Semantic Dot, 33,1 ở 64-NN, 47,1 ở phép sửa từ Ext-2 và **95,5** ở phép sửa từ Ext-1. Trong 93,23% truy vấn, phép sửa từ Ext-1 chọn ảnh sai có phân vị cao hơn Dot. Điều này ủng hộ liên hệ giữa lỗi và ngoại suy ở những hướng ít xuất hiện ở nguồn.

Để đối chiếu hiện tượng một số ảnh lặp lại trên nhiều truy vấn, 104 ảnh thêm có số lần xuất hiện nhiều nhất chiếm 20,57% số chỗ của ảnh thêm trong top 200 dưới Dot và 20,78% dưới phép sửa nguồn. Hai con số này chỉ đo mức tập trung ở ngưỡng 200. Chúng chưa xác định nguyên nhân của lỗi ở hạng nhất.

Đối chứng dương dùng toàn bộ 10.463 sketch thuộc 17 lớp bổ sung. Dưới phép sửa Ext-1, có 60 ảnh khác nhau đứng đầu sai cho các sketch của bốn lớp chung. Trong số đó, **40 ảnh** cũng đứng đầu đúng cho ít nhất một sketch của 17 lớp kia. Tính theo số lần xuất hiện, 40 ảnh này chiếm 94,38% các lần một ảnh bổ sung đứng đầu sai cho bốn lớp chung. Trong 4.849 lần phép sửa đứng đầu đúng trên 17 lớp kia, 59,64% đáp án đúng nằm trong 10% ảnh có leverage cao nhất của lớp chúng. Hạ điểm mọi ảnh ít được đại diện ở nguồn do đó tác động đến cả nhiều đáp án đúng. Tác động ròng lên mAP của một hình thức phạt cụ thể chưa được đo.

## Câu hỏi nghiên cứu còn mở

Với cùng một ảnh đích ít được đại diện ở gallery nguồn, bằng chứng nào trong **riêng cặp sketch–ảnh** cho biết khi nào việc tăng điểm đưa đúng đối tượng lên hạng và khi nào nó đẩy một đối tượng khác lên nhầm? Một can thiệp độc lập có thể kiểm chứng trên các cặp có ảnh đích trùng ID như trên, với cả hai chiều đúng/sai, cùng CLIP đóng băng và cùng evaluator. Tiêu chuẩn là giảm lỗi khi thêm ứng viên, giữ được truy hồi đúng cho chính những ảnh đó, rồi đạt mức tăng mAP với cấu hình đã khóa. Mượn đặc trưng từ láng giềng, biểu diễn toán tử đồ thị bằng một tích vô hướng và dùng leverage để đo độ phủ đều có tiền lệ trong các nghiên cứu dưới đây. Bằng chứng hiện tại xác định một tình huống lỗi của phép chuyển từ Ext-1. Hiện vẫn cần bằng chứng rằng cùng nguyên nhân chi phối các lỗi của CLIP gốc, và chưa có phương pháp độc lập nào trong thí nghiệm này đạt mục tiêu +4–5 trên các benchmark. TU-Berlin ở đây đo chuyển tập dữ liệu trong cùng chiều sketch → photo.

## Prior art nên đọc

1. **Lazaridou, Dinu và Baroni, ACL–IJCNLP 2015.** Trong dịch từ và gán nhãn ảnh zero-shot, tác giả phân tích hiện tượng một số đích hay được chọn cho nhiều đầu vào sau khi ánh xạ giữa hai không gian. Liên hệ trực tiếp là đo mức độ tập trung của các lỗi sau khi chuyển phép sửa sang gallery mới. Công trình dùng cặp dữ liệu để ước lượng ánh xạ. [Hubness and Pollution](https://aclanthology.org/P15-1027/)
2. **Pauwels, Bach và Vert, NeurIPS 2018.** Tác giả liên hệ statistical leverage với mật độ dưới các giả thiết về kernel và phân bố. Điều này gợi ý phép đo độ phủ của gallery nguồn trong không gian tọa độ từ. Giá trị leverage tuyến tính hữu hạn ở đây chỉ đo quan hệ với thiết kế nguồn. [Relating Leverage Scores and Density using Regularized Christoffel Functions](https://proceedings.neurips.cc/paper_files/paper/2018/hash/aff1621254f7c1be92f64550478c56e6-Abstract.html)
3. **Iscen và cộng sự, CVPR 2018.** Tác giả tính gần đúng cơ sở phổ của đồ thị gallery ở bước offline, rồi biểu diễn kết quả tìm kiếm trên manifold bằng embedding và dot product. Công trình này đặt ranh giới novelty cho ý tưởng viết lại phép lan truyền đồ thị thành một phép nhân vector. [Fast Spectral Ranking for Similarity Search](https://arxiv.org/abs/1703.06935)
4. **Efthymiadis và cộng sự, WACV 2025.** FreeDom lấy ảnh proxy theo từng query, tìm từ cho từng proxy rồi dùng các từ được nhiều proxy ủng hộ cùng tên miền đích để tìm ảnh. Đây là đối chứng CIR gần nhất cho hướng đổi biểu diễn truy vấn bằng từ và láng giềng. [FreeDom](https://cmp.felk.cvut.cz/~efthynik/freedom/freedom)

Hai đối chiếu khác trực tiếp giới hạn cách tuyên bố đóng góp. Arandjelović và Zisserman đã chỉ ra bằng ảnh thật rằng mượn visual word từ ảnh lân cận có thể tạo false positive, rồi kiểm tra vùng tương ứng hình học trên bài toán truy hồi cùng đối tượng. BASIC dùng các nguồn điểm riêng và late fusion trong CIR, với tập đánh giá có hard negative được chọn có chủ đích. [CVPR 2012, §5 và Hình 3](https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf) · [BASIC, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/6391a02f02dc529cfa3f242fb4343c6a-Abstract-Conference.html)

## Tái lập

Protocol chính và mã trước khi chạy có commit `2b1355a3c4296e116de57c07c300e1604a1d8779`. Phân tích hậu nghiệm Sketchy dùng `tf_sbir.semantic_operator_overlap` tại `b2da81af4f5d2834cc1dc36c58f73af62d034d67`. Cấu hình Kaggle có `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Trong checkout chứa cache frozen và evaluator KD-SBIR tương ứng, chạy `python -m tf_sbir.semantic_operator_audit prepare --split SPLIT --revision RUN_SHA` cho từng split, sau đó `score --split SPLIT --part 0` và `--part 1` với cùng `--revision`. Lệnh `report --revision RUN_SHA` đọc toàn bộ các shard đã xác minh. Lệnh `python -m tf_sbir.semantic_operator_overlap --revision OVERLAP_SHA` tạo bảng đối chiếu Sketchy từ các index đó. Nhãn chỉ tham gia phần đánh giá và thống kê hậu nghiệm.

Số liệu gốc và giá trị theo từng truy vấn có tại [operator-summary.json](operator-summary.json), [sketchy-overlap.json](sketchy-overlap.json) và [per-query-L4.zip](per-query-L4.zip). Tệp ZIP có SHA-256 `a537438bfe0ff0654d6afb5a32520987bdba5d3c1b2be96e945557a002eeb230`.
