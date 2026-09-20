# Individual contribution report

## Thông tin

- Họ và tên: Đào Ngọc Bình Thiên
- Mã học viên: 2A202602814
- Nhóm: K4-L3A (TV4 — Generation + UI + Evaluation)
- Repository/branch: https://github.com/tsun165/K4-L3A-RAG-Pipeline — nhánh `thiendao103` (PR #2 vào `main`)

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| Golden Dataset | Soạn 16 test case Q&A có grounded context chuẩn xác 100% từ 3 văn bản quy phạm pháp luật (Luật 73/2021/QH14, Nghị định 144/2021, Nghị định 105/2021) và 5 bài báo; đáp ứng cấu trúc `{question, expected_answer, expected_context}` | `group_project/evaluation/golden_dataset.json` @ `bd2b5ae` | Done |
| Streamlit UI | Thiết kế giao diện 2 cột độc lập (Chat Console bên trái & Citations Inspector bên phải); khung chat có thanh cuộn riêng `height=560` cố định, ô nhập liệu không bị trôi; hiển thị thẻ nguồn (Luật/Tin tức), điểm tương đồng (Score %), trích đoạn căn cứ và link bài viết gốc | `app.py` @ `bd2b5ae`, `bca2d0c` | Done |
| Trải nghiệm người dùng (UX) | Tích hợp hiệu ứng tìm kiếm gọn gàng (`st.spinner`) và hiệu ứng xuất chữ từng từ (typewriter/streaming cadence 0.03s) mô phỏng chatbot AI phản hồi trực quan; bổ sung các nút câu hỏi mẫu và thanh trượt cấu hình retrieval | `app.py` @ `bca2d0c` | Done |
| Nối Backend thật vào UI | Kết nối trực tiếp hàm `generate_with_citation()` của Task 10 vào `app.py`, loại bỏ hoàn toàn mã mock; đồng bộ luồng truy vấn: Dense + BM25 ➔ RRF ➔ Reorder ➔ LLM Call ➔ Citation Inspector | `app.py` @ `bca2d0c` | Done |
| Task 10 — Context Reordering | Cài đặt `reorder_for_llm()` chống hiện tượng *lost-in-the-middle* theo kỹ thuật xen kẽ `front + back[::-1]`, bảo đảm không làm biến đổi (non-mutating) danh sách gốc và giữ nguyên 100% ID | `src/task10_generation.py` @ `bca2d0c` | Done |
| Task 10 — Context Formatting | Cài đặt `format_context()` tự động trích xuất `title`, `source` và `content` thành cấu trúc `[Document i | Title: ... | Source: ...]`, chuẩn bị sẵn sàng cho LLM tạo citation kiểm chứng được | `src/task10_generation.py` @ `bca2d0c` | Done |
| Task 10 — Dispatch & Refusal | Xây dựng hàm `call_llm()` dispatch linh hoạt theo provider (OpenAI, Gemini, Anthropic), cấu hình system prompt ép trích dẫn nguồn và xử lý từ chối an toàn (*Safe Refusal*) khi câu hỏi ngoài tầm tri thức | `src/task10_generation.py` @ `bca2d0c` | Done |
| Báo cáo đánh giá (Evaluation) | Chủ trì hoàn thiện toàn bộ báo cáo đánh giá: đo đạc 4 metrics Ragas, so sánh A/B giữa Config A (Dense-only) và Config B (Hybrid+RRF), phân tích 3 ca lỗi (Worst performers) và đề xuất cải tiến | `group_project/evaluation/RESULT.md`, `reports/RESULT.md` | Done |

## Quyết định kỹ thuật quan trọng

1. **Quyết định:** Thiết kế giao diện Streamlit theo bố cục 2 cột song song (Chat bên trái, Citations Inspector bên phải) với thanh cuộn độc lập (`st.container(height=560)`), đồng thời tự động đồng bộ nguồn trích dẫn từ tin nhắn trợ lý gần nhất.
   **Lý do/evidence:** Khi số lượng câu hỏi và câu trả lời tăng lên, giao diện mặc định của Streamlit sẽ bị kéo dài vô tận, khiến ô chat input bị trôi xuống đáy màn hình và người dùng phải cuộn chuột liên tục. Ngoài ra, việc tách riêng cột trích dẫn giúp người dùng luôn đối soát được văn bản luật/bài báo căn cứ ngay lập tức mà không làm rối mắt khung trò chuyện. Việc truy xuất nguồn từ `last_assistant_msg` giải quyết triệt để lỗi panel trích dẫn bị rỗng khi trang web rerender.
   **Trade-off:** Đòi hỏi cấu hình CSS nâng cao và tính toán chiều cao hợp lý để tương thích cả Dark Mode lẫn Light Mode; giao diện tối ưu nhất trên màn hình máy tính (Wide Layout), hạn chế hơn trên thiết bị di động hẹp.

2. **Quyết định:** Áp dụng chiến lược Reorder context xen kẽ hai đầu (`front = chunks[::2]`, `back = chunks[1::2]`, trả về `front + back[::-1]`) trước khi nạp vào prompt cho LLM.
   **Lý do/evidence:** Hiện tượng *Lost in the Middle* (Liu et al.) chứng minh các mô hình ngôn ngữ lớn (LLM) thường ghi nhớ và chú ý tốt nhất ở phần đầu và phần cuối của ngữ cảnh nạp vào, trong khi các đoạn văn bản ở giữa dễ bị bỏ quên. Bằng cách đưa các chunk có điểm xếp hạng cao nhất ra hai đầu, LLM sẽ nắm bắt bằng chứng tốt hơn, giảm thiểu tình trạng bỏ sót thông tin và hallucination khi tổng hợp câu trả lời.
   **Trade-off:** Thứ tự các đoạn văn bản gửi cho LLM không còn giữ nguyên thứ tự giảm dần tuyệt đối như ban đầu; cần ánh xạ cẩn thận để số thứ tự `[Nguồn i]` trong câu trả lời vẫn đối chiếu chính xác với danh sách `sources` trả về cho UI.

## Kiểm thử và kết quả

- Test tự động (Automated Tests):
  - Chạy toàn bộ test suite dự án bằng `pytest -v` đạt kết quả tuyệt đối: **31/31 passed 100%**.
  - `pytest tests/test_acceptance.py -q` ➔ **5 passed**:
    - `test_golden_dataset_has_15_grounded_cases`: Pass (16 Q&A grounded, đủ 3 trường bắt buộc).
    - `test_evaluation_report_is_completed`: Pass (xác nhận `RESULT.md` không còn chữ `TODO`, đủ 4 đề mục chuẩn).
  - `pytest tests/test_contracts.py -q` ➔ **15 passed**:
    - `test_reorder_is_non_mutating_and_context_contains_source`: Pass.
    - `test_generation_result_validator_accepts_safe_refusal`: Pass.
    - `test_public_function_signatures_are_stable`: Pass.
  - `pytest tests/test_retrieval.py -q` ➔ **11 passed**.
- Kiểm thử thực tế qua UI (`streamlit run app.py` tại `http://localhost:8501`):
  - Kịch bản 1 (Văn bản Luật): Query *"Mức phạt hành chính đối với hành vi sử dụng ma túy?"* ➔ Trả lời chính xác mức phạt 1–2 triệu đồng theo Điều 23 NĐ 144, hiển thị thẻ xanh `VĂN BẢN LUẬT` và điểm tương đồng 95%.
  - Kịch bản 2 (Tin tức): Query *"Ca sĩ Miu Lê bị bắt vì hành vi gì và ở đâu?"* ➔ Trả lời đúng tội danh tại Cát Bà (Hải Phòng), thẻ xanh `TIN TỨC / BÀI BÁO` và kèm liên kết bài viết gốc.
  - Kịch bản 3 (Ngoài domain): Query *"Thời tiết Hà Nội hôm nay thế nào?"* ➔ Trả về phản hồi từ chối an toàn (*Safe Refusal*), không bịa đặt thông tin và cảnh báo không có nguồn trích dẫn.
- Lỗi đã phát hiện và cách xử lý:
  - Lỗi panel trích dẫn bị rỗng khi click câu hỏi mẫu: Sửa cơ chế lưu session state, tự động bóc tách trích dẫn từ tin nhắn phản hồi gần nhất của assistant.
  - Lỗi giao diện bị cuộn trôi: Bọc nội dung hội thoại và cột trích dẫn vào container cố định có thanh cuộn riêng.

## Điều còn hạn chế

- Giao diện hiện tại hiển thị trích dẫn dưới dạng khối văn bản Markdown trích xuất tĩnh; chưa hỗ trợ tính năng làm nổi bật trực tiếp dòng văn bản liên quan (Source / Citation Highlighting) trong tài liệu PDF gốc hoặc liên kết neo trực tiếp tới dòng văn bản.
- Mô hình nhúng đa ngôn ngữ `BAAI/bge-m3` có dung lượng khá lớn (~2.27 GB), gây nghẽn đường truyền khi tải lần đầu từ Hugging Face Hub trên môi trường mạng hạn chế nếu thiếu token xác thực.
- Nếu có thêm thời gian, thay đổi đầu tiên tôi sẽ thực hiện: Triển khai tính năng **Citation Highlighting** (tô màu câu văn làm bằng chứng khi người dùng rê chuột vào ký hiệu trích dẫn trên câu trả lời) để đạt điểm thưởng tối đa cho hạng mục UI theo thang điểm Rubric.

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- Ngày: 2026-09-20
- Tên thành viên: Đào Ngọc Bình Thiên
