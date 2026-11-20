# Hiệu hai vector kỳ vọng trong gallery

Cùng công thức, mỗi ảnh nguyên khung và toàn bộ 20.931 từ nguyên văn cho kết quả sau. Mỗi ô là precision / mAP (%).

| Phương pháp | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin |
|---|---:|---:|---:|
| CLIP gốc | 41,64 / 30,29 | 33,85 / 36,34 | 46,26 / 29,07 |
| Hiệu vector kỳ vọng có điều kiện và kỳ vọng đều | 78,30 / 75,59 | 64,56 / 65,91 | 75,70 / 68,41 |

Đây là một kết quả của phương pháp không cần biết tên miền và không dò tham số. Nó cải thiện trên cả hai bộ dữ liệu so với CLIP gốc. Khi chấm hai công thức đã định trước trên toàn bộ Sketchy, phương pháp likelihood khái niệm có trung bình precision cao hơn và được chọn trước lượt TU. [Giao thức](CONDITIONAL_INDEX_PROTOCOL_VI.md) · [Ablation trên Sketchy](results/conditional-semantic-index/sketchy-complete.json) · [Ablation trên TU](results/conditional-semantic-index/tuberlin-complete.json)
