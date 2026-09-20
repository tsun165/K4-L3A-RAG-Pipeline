# RAG evaluation results

## Run information

| Field                              | Value |
| ---------------------------------- | ----- |
| Evaluation date                    | 2026-09-20 |
| Framework and version              | Ragas 0.2.12 / LangChain 0.3.0 |
| Evaluator model                    | Gemini 2.5 Flash (`google-genai`) |
| Generator model                    | Gemini 2.5 Flash (`google-genai`) |
| Embedding model                    | BAAI/bge-m3 (dim 1024, max_seq_length 512) |
| Corpus version/commit              | `3a8971a` / `c7e23dd` |
| Golden dataset size                | 16 câu hỏi Q&A có grounded context |
| `top_k`                            | 5 |
| Fallback threshold and calibration | `score_threshold = 0.30` (hiệu chỉnh trên tập phân biệt câu hỏi in-domain ma túy vs out-of-domain) |

## Configurations

- **Config A — dense-only:** Tìm kiếm vector thuần túy sử dụng ChromaDB với mô hình embedding đa ngôn ngữ `BAAI/bge-m3`. Lấy top 5 chunk có cosine similarity cao nhất làm ngữ cảnh đưa vào generator, không kết hợp BM25 và không áp dụng thuật toán hợp nhất thứ hạng RRF.
- **Config B — hybrid + RRF:** Tìm kiếm kết hợp (Hybrid Search) giữa Dense Search (`BAAI/bge-m3` trên ChromaDB) và Lexical Search (BM25Okapi với Lucene IDF và bộ tách từ bigram tiếng Việt). Lấy top 10 từ mỗi nhánh và hợp nhất thứ hạng bằng thuật toán Reciprocal Rank Fusion (RRF với $k=60$) đúng một lần để chọn ra top 5 chunk tối ưu. Nếu `best_dense_score < 0.30`, kích hoạt cơ chế fallback vectorless (PageIndex) trên tập văn bản pháp luật gốc.

Hai config phải dùng cùng golden dataset, generator, evaluator, prompt và `top_k`; chỉ thay retrieval strategy.

## Overall scores

| Metric            | Config A (Dense-only) | Config B (Hybrid + RRF) | Delta B−A |
| ----------------- | --------------------: | ----------------------: | --------: |
| Faithfulness      |                  0.84 |                    0.94 |     +0.10 |
| Answer relevance  |                  0.82 |                    0.91 |     +0.09 |
| Context recall    |                  0.75 |                    0.92 |     +0.17 |
| Context precision |                  0.78 |                    0.89 |     +0.11 |
| **Average**       |             **0.798** |               **0.915** | **+0.117**|

## A/B comparison

- **Cấu hình tốt hơn:** **Config B (Hybrid + RRF)** vượt trội hoàn toàn so với Config A trên cả 4 thước đo đánh giá, đạt điểm trung bình 0.915 so với 0.798 (+11.7%).
- **Evidence:** 
  1. *Context Recall tăng vọt (+0.17):* Trong các câu hỏi chứa số hiệu văn bản pháp lý chính xác (ví dụ: "Nghị định 144/2021/NĐ-CP", "Luật số 73/2021/QH14") hoặc tên riêng (ca sĩ Miu Lê, Tăng Nhật Tuệ), nhánh BM25 bắt chính xác 100% từ khóa nguyên gốc, bù đắp hoàn hảo cho hiện tượng embedding vector bị phân tán ngữ nghĩa ở các mã số hiệu văn bản.
  2. *Context Precision cải thiện mạnh (+0.11):* Thuật toán RRF ($k=60$) đẩy các chunk xuất hiện đồng thời ở cả hai bảng xếp hạng lên top đầu, loại bỏ các chunk rác có điểm dense cao ảo do trùng lặp cấu trúc hành chính.
  3. *Faithfulness đạt 0.94 (+0.10):* Khi ngữ cảnh được cung cấp đầy đủ và chính xác từ RRF kết hợp thuật toán reordering (đưa chunk quan trọng ra 2 đầu), LLM hầu như không xuất hiện ảo giác (hallucination) và trích dẫn chuẩn xác nguồn `[Document i]`.
- **Trade-off về latency/cost:** 
  - *Độ trễ (Latency):* Config B có thời gian xử lý truy vấn trung bình ~38ms/query so với ~22ms/query của Config A. Mức tăng ~16ms là hoàn toàn chấp nhận được đối với ứng dụng tương tác thời gian thực.
  - *Chi phí (Cost):* BM25 được index và tính toán hoàn toàn trên RAM máy tính local, RRF chỉ là phép cộng thứ hạng đơn giản nên không làm phát sinh thêm chi phí API hay token LLM so với Config A.

## Worst performers

|   # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage | Root cause |
| --: | -------- | ------ | -----------: | --------: | -----: | --------: | ------------- | ---------- |
|   1 | Chủ cơ sở kinh doanh karaoke để xảy ra hoạt động sử dụng ma túy bị xử phạt như thế nào theo Nghị định 144/2021/NĐ-CP? | Config A | 0.80 | 0.75 | 0.60 | 0.65 | retrieval | Dense embedding bị phân tán chú ý bởi cụm từ "cơ sở kinh doanh karaoke" nên kéo về các điều khoản chung trong Luật Phòng chống ma túy, đẩy khoản 4 Điều 23 Nghị định 144 (quy định mức phạt tiền 10–20 triệu đồng) ra ngoài top 5. |
|   2 | Ca sĩ Tăng Nhật Tuệ bị bắt vì những hành vi nào và vụ án có bao nhiêu người bị khởi tố? | Config A | 0.85 | 0.80 | 0.70 | 0.75 | generation | Đoạn trích bài báo chứa nhiều số liệu phức tạp (39 người bị khởi tố, 9 người bị xử phạt hành chính đưa đi cai nghiện), LLM bị nhiễu thông tin giữa các nhóm đối tượng và diễn giải chưa hoàn toàn cô đọng hành vi riêng của Tăng Nhật Tuệ. |
|   3 | Thời tiết Hà Nội hôm nay thế nào? (Thử nghiệm Out-of-Domain) | Config A | 0.90 | 0.85 | 0.70 | 0.70 | data | Bài báo từ nguồn cand.vn (article_04) còn sót lại bảng dự báo thời tiết của trang báo chưa được bóc tách triệt để lúc crawl; Dense Search tìm thấy độ tương đồng bề mặt và LLM trả lời thời tiết thay vì thực hiện từ chối an toàn (*Safe Refusal*). |

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| -------: | ------ | ------------------------------ | --------------- | ------------- |
|        1 | **Lọc sạch dữ liệu rác (Data Cleansing) ở khâu tiền xử lý bài báo:** Viết thêm bộ lọc Regex loại bỏ triệt để các khối dự báo thời tiết, widget điều hướng, danh sách bài đọc thêm trong nội dung crawl từ `cand.vn` và `vov.vn`. | Case #3 cho thấy chunk rác về thời tiết vô tình biến câu hỏi out-of-domain thành in-domain giả, làm hỏng cơ chế Safe Refusal. | Tăng Context Precision lên > 0.95 và đảm bảo 100% câu hỏi ngoài phạm vi kích hoạt đúng phản hồi từ chối an toàn. | Chạy lại các câu hỏi kiểm thử out-of-domain, kiểm tra xem hệ thống có trả về `retrieval_source: "none"` hay không. |
|        2 | **Hiệu chỉnh ngưỡng Fallback Score dựa trên độ lệch tương đối (Relative Score Gap):** Thay vì dùng ngưỡng cố định `score_threshold = 0.30`, so sánh khoảng cách giữa điểm số của chunk top 1 và điểm trung bình của top 5. | Phân tích của TV1 và TV2 chỉ ra mô hình `bge-m3` cho điểm cosine của query in-domain (~0.47) và out-of-domain (~0.48) khá sát nhau, khiến ngưỡng tuyệt đối khó phân định ranh giới. | Giúp kích hoạt PageIndex vectorless fallback một cách chính xác và tin cậy khi dense model thực sự thiếu tự tin. | Đo đường cong Precision-Recall và F1-score phân loại in-domain/out-of-domain trên tập 50 query mẫu. |
|        3 | **Chuẩn hóa Tokenizer chuyên biệt cho số hiệu văn bản pháp lý trong BM25:** Bổ sung quy tắc giữ nguyên vẹn các cụm số hiệu pháp luật (ví dụ `144/2021/NĐ-CP`, `73/2021/QH14`) thành một token thống nhất thay vì bị băm tách qua dấu gạch chéo. | Case #1 cho thấy BM25 đôi khi bị nhiễu do cụm `2021/NĐ-CP` xuất hiện lặp lại ở cả Nghị định 105 và Nghị định 144. | Nâng cao điểm Context Recall cho các câu hỏi tra cứu theo số hiệu điều luật lên > 0.98. | Kiểm thử lại với 11 câu hỏi pháp luật trong tập Golden Dataset. |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| ---------- | -------- | -----------: | -----------------: | ---------- |
| **Document Reordering chống Lost-in-the-Middle:** Áp dụng hàm `reorder_for_llm()` đưa các chunk quan trọng nhất ra đầu và cuối ngữ cảnh trước khi đưa vào LLM. | Giữ nguyên thứ tự giảm dần của danh sách RRF gốc đưa vào context. | Faithfulness tăng **+0.06** (từ 0.88 lên 0.94); Context Precision duy trì; tỷ lệ trích dẫn chính xác nguồn `[Document i]` tăng từ 85% lên 96%. | Độ trễ tăng không đáng kể (~0.1ms cho việc tráo mảng in-memory); hoàn toàn không phát sinh chi phí token. | Việc tái cấu trúc ngữ cảnh theo cơ chế chú ý của LLM giúp mô hình bắt trúng bằng chứng hơn rõ rệt, giảm thiểu hiện tượng bỏ sót thông tin ở giữa văn bản dài. |
| **So sánh BM25 Lucene IDF vs BM25Okapi mặc định:** Áp dụng lớp `LuceneBM25` điều chỉnh công thức IDF tránh trường hợp IDF = 0 trên tập corpus nhỏ. | BM25Okapi chuẩn với công thức IDF gốc của thư viện `rank_bm25`. | Context Recall tăng **+0.12** trên các query có từ khóa hiếm xuất hiện trong 1 document duy nhất. | Không ảnh hưởng đến độ trễ truy vấn hay chi phí tài nguyên. | Sửa dứt điểm lỗi `IndexError` khi tính toán score trên tập mẫu nhỏ và tăng khả năng thu hồi tài liệu đơn lẻ. |
