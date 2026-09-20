# Individual contribution report — TV3

- **Họ và tên:** Nguyễn Nguyên Phong
- **Mã học viên:** 2A202602691
- **Nhóm:** K4-L3A-RAG-Pipeline
- **Vai trò:** TV3 — Fusion + Fallback + PageIndex
- **Repository/branch:** `contrib/Heargreaves1` (target PR: `tsun165/K4-L3A-RAG-Pipeline:main`)

---

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| **Task 8: PageIndex vectorless fallback** | Cài đặt `pageindex_search()` với cơ chế vectorless fallback trên văn bản pháp luật, chống crash khi dịch vụ ngoài offline, chuẩn hóa `SearchResult` contract. | `src/task8_pageindex_vectorless.py` | **Done** |
| **Task 9: Retrieval Pipeline & Fallback** | Tích hợp Dense + Sparse, điều phối RRF đúng 1 lần, so sánh `best_dense_score` với `score_threshold` để kích hoạt PageIndex fallback, xử lý lỗi ngoại lệ an toàn. | `src/task9_retrieval_pipeline.py` | **Done** |
| **Contract tests** | Viết và kiểm thử các kịch bản fallback, test dense confident và provider unavailable, đảm bảo 100% contract test pass. | `tests/test_contracts.py` | **Done** |

---

## Quyết định kỹ thuật quan trọng

1. **Quyết định:** Sử dụng Cosine Score gốc của Dense Search để quyết định ngưỡng Fallback (không dùng RRF score).
   - **Lý do/evidence:** RRF (Reciprocal Rank Fusion) chỉ biểu diễn thứ hạng tương đối ($\sum \frac{1}{k+r}$), điểm số bị nén trong khoảng $[0.01, 0.04]$ và không phản ánh độ tin cậy thực tế của ngữ nghĩa. Cosine score phản ánh trực tiếp khoảng cách vector trong không gian embedding; nếu `best_dense_score < score_threshold` (ví dụ 0.3) tức là câu hỏi nằm ngoài vùng tri thức của corpus, cần kích hoạt PageIndex vectorless fallback.
   - **Trade-off:** Cần hiệu chỉnh (calibrate) ngưỡng `score_threshold` theo từng mô hình embedding cụ thể.

2. **Quyết định:** Thiết kế cơ chế Fallback phòng thủ 2 lớp (Fault-Tolerant Fallback) cho Task 8 & 9.
   - **Lý do/evidence:** PageIndex là dịch vụ API bên ngoài, dễ gặp rủi ro timeout, hết quota hoặc mất mạng. Nếu API lỗi, Task 8 tự động chuyển sang quét văn bản trực tiếp (vectorless text scan) trên thư mục văn bản pháp luật; đồng thời Task 9 bọc `try...except` để nếu fallback lỗi thì trả về kết quả Hybrid thay vì làm sập (crash) ứng dụng của người dùng.
   - **Trade-off:** Cần xử lý logic dự phòng phức tạp hơn nhưng đổi lại độ sẵn sàng (high availability) của hệ thống đạt mức tối đa.

---

## Kiểm thử và kết quả

- **Test đã chạy:**
  - `test_retrieve_uses_dense_score_for_fallback`: Kiểm tra khi dense score thấp (< threshold) hệ thống tự động gọi PageIndex fallback.
  - `test_retrieve_fuses_once_when_dense_is_confident`: Kiểm tra khi dense score cao (tin cậy), chỉ fuse RRF đúng 1 lần và không gọi fallback thừa thãi.
  - `test_retrieve_survives_fallback_provider_error`: Kiểm tra hệ thống vẫn hoạt động an toàn khi provider ném `RuntimeError`.
- **Kết quả:** Vượt qua toàn bộ các test contract (`PASSED 100%`).

---

## Điều còn hạn chế

- **Hạn chế:** Ngưỡng `score_threshold = 0.3` hiện tại đang được đặt theo giá trị tham chiếu chung, cần chạy thêm các query out-of-domain để tìm ra ngưỡng tối ưu nhất cho corpus ma túy/pháp luật.
- **Nếu có thêm thời gian:** Tôi sẽ xây dựng một script tự động hiệu chỉnh (grid-search calibration) để tìm ra điểm cân bằng ngưỡng F1-score tối ưu giữa Dense và PageIndex fallback.

---

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- **Ngày:** 20/09/2026
- **Tên thành viên:** Nguyễn Nguyên Phong
