# Individual contribution report

## Thông tin

- Họ và tên: Nguyễn Vũ Huy
- Mã học viên: 2A202602662
- Nhóm: K4-L3A (TV2 — Retrieval: Dense + Lexical)
- Repository/branch: https://github.com/tsun165/K4-L3A-RAG-Pipeline — nhánh `feature/tv2-retrieval`, đã merge vào `main` (code tại `2011cec`, report tại `ea389a6` và bản cập nhật sau)

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| Task 5 — Dense search | `semantic_search()`: dùng chung `embed_texts()`/`get_collection()` của Task 4, đổi cosine distance của Chroma thành similarity kẹp về [0, 1], giới hạn `n_results` theo `collection.count()` để index nhỏ không lỗi | `src/task5_semantic_search.py` @ `2011cec` | Done |
| Task 6 — BM25 lexical search | `lexical_search()` trên cùng corpus chunk nạp từ ChromaDB (cùng id với dense); tokenizer tiếng Việt âm tiết + bigram; IDF kiểu Lucene; cache index, tự rebuild khi corpus đổi | `src/task6_lexical_search.py` @ `2011cec` | Done |
| Task 7 — RRF | `rerank_rrf()`: `sum(1/(k+rank))`, rank từ 1, dedupe trong từng list, tie-break ổn định, không mutate input, gắn `retrieval_method="hybrid"` | `src/task7_reranking.py` @ `2011cec` | Done |
| Helper chung | Chuẩn hoá metadata đọc lại từ Chroma (`url ""` → `None`, `chunk_index` → `int`) để mọi SearchResult qua được validator | `src/retrieval_utils.py` @ `2011cec` | Done |
| Unit test | 11 test offline cho Task 5–7 (tokenizer, cache, kẹp score, RRF tie/dedupe/non-mutating) | `tests/test_retrieval.py` @ `2011cec` | Done |
| Tích hợp với Task 4 (TV1) | Chạy end-to-end Task 4 → 5 → 6 → 7 trên Chroma thật với embedding giả; kiểm tra idempotent, round-trip metadata, id khớp giữa dense và BM25 | smoke script (không commit) | Done |
| Chạy với bge-m3 thật | Index 1274 chunk bằng bge-m3 trên CPU (download 2,27 GB bị nghẽn mạng, phải viết downloader song song có resume); đo cosine thật: in-domain 0,65–0,81, out-of-domain 0,38–0,63; dense query 150–390 ms; đề xuất TV3 nâng `SCORE_THRESHOLD` từ 0,3 lên ≈0,63 | `chroma_db/` local (gitignore), số liệu trong mục Kiểm thử | Done |

## Quyết định kỹ thuật quan trọng

1. **Quyết định:** Override IDF của `rank_bm25.BM25Okapi` bằng công thức Lucene `log(1 + (N − n + 0.5)/(n + 0.5))` (lớp `LuceneBM25`).
   **Lý do/evidence:** IDF gốc `log((N − n + 0.5)/(n + 0.5))` bằng 0 khi một từ xuất hiện ở đúng 1 trong 2 doc, nên với corpus 2 doc trong `test_lexical_search_returns_bm25_contract` mọi score = 0, bị lọc hết và test rớt (`IndexError`). Đã tái hiện: `idf['tuition'] = 0.0`, `scores = [0, 0]`. Sau khi đổi: test pass.
   **Trade-off:** Phụ thuộc vào method nội bộ `_calc_idf` của thư viện; score BM25 không so sánh trực tiếp được với nhóm khác dùng IDF mặc định (không ảnh hưởng RRF vì RRF chỉ dùng rank).

2. **Quyết định:** Tokenizer tiếng Việt = tách âm tiết (unicode `\w+`, lowercase, NFC) + bigram âm tiết liền kề (`"ma túy"` → `ma`, `túy`, `ma_túy`), thay vì thêm pyvi/underthesea.
   **Lý do/evidence:** Không phải sửa `pyproject.toml` dùng chung của nhóm; bigram giúp BM25 khớp từ ghép và số hiệu văn bản (`73_2021`); build index 1274 chunk mất 0,08 s, mỗi query 1–5 ms. Query "cai nghiện ma túy tự nguyện tại gia đình" trả đúng Điều 30 Luật 73/2021.
   **Trade-off:** Số token gấp đôi (tốn bộ nhớ hơn); bigram bắc qua dấu câu tạo nhiễu; kém chính xác hơn tách từ thật ở các cụm nhập nhằng.

## Kiểm thử và kết quả

- Test hoặc query tôi đã dùng: `pytest tests/test_contracts.py tests/test_retrieval.py -q` (3 contract test Task 5–7 + 11 unit test đều pass; các test fail còn lại thuộc Task 9–10 chưa implement). Smoke test trên corpus thật 1274 chunk với các query "Miu Lê bị bắt", "Tăng Nhật Tuệ", "cai nghiện ma túy tự nguyện tại gia đình", "Nghị định 144/2021/NĐ-CP phạt tiền…", và một query ngoài domain.
- Kết quả trước/sau nếu có: Trước khi đổi IDF, BM25 trả 0 kết quả trên contract test; sau khi đổi trả đúng `chunk-0` đầu tiên. End-to-end với Task 4 thật: index 2 lần vẫn 1274 chunk (không trùng), `url` legal = `None`, `url` news = str, 5–6/10 id trùng nhau giữa dense và BM25 nên RRF fuse được. Với bge-m3 thật (5 query in-domain, 4 query out-of-domain): cosine tốt nhất in-domain 0,650–0,812, out-of-domain 0,375–0,625, khoảng cách phân tách hẹp (0,625 vs 0,650) nên threshold nên đặt ≈0,63 và cần thêm query để hiệu chỉnh; với threshold 0,3 hiện tại fallback không bao giờ chạy. `retrieve()` của TV3 trả hybrid trong 130–160 ms. Số liệu A/B dense-only vs hybrid + RRF trên golden dataset xem `group_project/evaluation/RESULT.md` (TV4 chủ trì).
- Lỗi đã phát hiện và cách xử lý: (1) IDF = 0 trong code gợi ý của template → `LuceneBM25`. (2) Chroma không lưu được `None` nên TV1 lưu `url=""` → `normalize_metadata()` khôi phục `None`. (3) Chroma lỗi/cảnh báo khi `n_results` lớn hơn số phần tử → chặn theo `count()`. (4) News chunk chứa nhiều link điều hướng của trang báo → đã báo TV1 lọc ở Task 2–3.

## Điều còn hạn chế

- Một hạn chế cụ thể của phần tôi làm: query theo số hiệu văn bản ("Nghị định 144/2021/NĐ-CP") bị BM25 xếp chunk boilerplate của Nghị định 105 lên đầu vì cụm `2021/NĐ-CP` lặp lại ở nhiều chỗ; hiện trông cậy vào dense + RRF để bù.
- Nếu có thêm thời gian, thay đổi đầu tiên tôi sẽ thực hiện: lọc boilerplate/stopword pháp lý trước khi index BM25 và A/B với tách từ bằng pyvi; sau đó thử cross-encoder reranker so với RRF để lấy bonus.

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- Ngày: 20/09/2026
- Tên thành viên: Nguyễn Vũ Huy
