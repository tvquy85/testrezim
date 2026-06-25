# AGENTS.md
Lập kế hoạch

Mọi plan phải dựa trên:

- Context trước đó trong trao đổi.
- Phân tích và kết quả đã có ở bước trước.

Khi đề xuất bước tiếp theo, phải dựa trên phân tích trước đó, không làm rời rạc hoặc cảm tính.

Luôn ưu tiên giải pháp tổng quát, có thể tái sử dụng và mở rộng. Không xử lý theo kiểu ad hoc nếu chưa có lý do kỹ thuật rõ ràng.

Điều tra và tìm giải pháp

Khi gặp lỗi, kết quả yếu hoặc hiện tượng bất thường, phải xử lý với vai trò **chuyên gia ML hàng đầu**:

- Investigate kỹ nguyên nhân.
- Tìm và đối chiếu với nguồn uy tín.
- Ưu tiên paper, benchmark, tài liệu kỹ thuật chính thống hoặc tài liệu từ nguồn đáng tin cậy.
- Không kết luận vội nếu chưa đủ bằng chứng.

Giải pháp đưa ra phải có cơ sở kỹ thuật, có khả năng tổng quát hóa và phù hợp với mục tiêu paper.

Góc nhìn đánh giá

Luôn đánh giá từ hai vai trò:

- **Principal ML Scientist**: tập trung vào mô hình, dữ liệu, phương pháp, độ tin cậy và khả năng tổng quát.

Chạy thí nghiệm

Tận dụng GPU **RTX 3090**, nhưng phải kiểm soát thời gian và tài nguyên.

Với thí nghiệm mới hoặc có khả năng chạy lâu:

- Bắt đầu từ scale nhỏ.
- Mở rộng dần lên scale trung bình/lớn.
- Mỗi stage có `run_id` riêng.
- Mỗi stage có audit riêng.
- Chỉ sang stage tiếp theo khi pass gate đã định nghĩa.

Stage gợi ý:

```text
stage_0_sanity_check
stage_1_small_scale
stage_2_medium_scale
stage_3_full_scale
```


Thư viện và môi trường

Khi thiếu thư viện, kiểm tra theo thứ tự:

1. Môi trường hiện tại hoặc thư mục `d:\Conferences\NIPS\FinEval_Prev\FinEval\`.
2. `d:\LOBProj\LOBExp\`.
3. Download từ Internet khi hai nguồn trên không có.

Không cài đặt tùy tiện nếu có thể tái sử dụng môi trường, thư viện hoặc cache sẵn có.
