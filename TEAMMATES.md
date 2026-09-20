# Phân công công việc — K4-L3A-RAG-Pipeline

Tài liệu này chia việc cho 4 thành viên theo **cụm chạy song song**, để mỗi người có thể `git clone` và làm việc độc lập, ít conflict nhất có thể. Xem chi tiết interface bắt buộc ở [docs/MODULE_CONTRACTS.md](docs/MODULE_CONTRACTS.md) trước khi code.

## Quy ước làm việc

```bash
git clone https://github.com/tsun165/K4-L3A-RAG-Pipeline.git
cd K4-L3A-RAG-Pipeline
git checkout -b feature/<ten-nhanh-cua-ban>
# ... code ...
git push -u origin feature/<ten-nhanh-cua-ban>
# Mở PR vào main, tag người liên quan để review
```

- Mỗi người **chỉ sửa file mình sở hữu** ở Cụm 1–2 để tránh conflict. Từ Cụm 3 trở đi các module bắt đầu phụ thuộc nhau — pull `main` mới nhất trước khi bắt đầu.
- Trước khi mở PR: chạy `pytest tests/test_contracts.py -q` cho đúng phần mình làm.
- Không commit `.env` hoặc API key thật.

## Thành viên

| | Thành viên | Mã học viên | Vai trò chính |
|---|---|---|---|
| TV1 | Đỗ Thái Sơn | 2A202603021 | Data pipeline + Indexing |
| TV2 | Nguyễn Vũ Huy | 2A202602662 | Retrieval (Dense + Lexical) |
| TV3 | | | Fusion + Fallback + PageIndex |
| TV4 | Đào Ngọc Bình Thiên | 2A202602814 | Generation + UI + Evaluation |

---

## Cụm 1 — Chạy song song ngay từ đầu (không phụ thuộc nhau)

Cả 4 người có thể bắt đầu **cùng lúc**, không phải chờ nhau.

| Thành viên | Việc | File | Ghi chú |
|---|---|---|---|
| **TV1** | Task 1–3: thu thập + chuẩn hóa dữ liệu | `task1_collect_legal_docs.py`, `task2_crawl_news.py`, `task3_convert_markdown.py` | ✅ Đã xong — output có sẵn ở `data/standardized/` cho cả nhóm dùng |
| **TV1** | Task 4: chunk + embed + index ChromaDB | `task4_chunking_indexing.py` | Việc khóa (blocking) — ưu tiên làm ngay sau khi clone, để mở đường cho TV2 |
| **TV3** | Task 8: PageIndex vectorless fallback | `task8_pageindex_vectorless.py` | Độc lập hoàn toàn — đọc trực tiếp PDF gốc ở `data/landing/legal/`, không cần chờ Task4 |
| **TV4** | Soạn Golden dataset (tối thiểu 15 câu hỏi) | `group_project/evaluation/golden_dataset.json` | Chỉ cần đọc corpus ở `data/standardized/`, không cần chờ pipeline chạy được |
| **TV4** | Khung Streamlit UI (dữ liệu giả lập) | `app.py` | Dựng layout: ô hỏi, khung answer/source/score — dùng mock `GenerationResult` tạm thời |
| **TV2** | Chuẩn bị: đọc contract Task5/6, viết test stub, chọn tokenizer BM25 tiếng Việt | `task6_lexical_search.py` (khung sườn) | Chưa code logic thật vì cần chunks thật từ Task4 |

---

## Cụm 2 — Sau khi Task4 (TV1) merge vào main

| Thành viên | Việc | File | Phụ thuộc |
|---|---|---|---|
| **TV2** | Task 5: dense/semantic search trên ChromaDB | `task5_semantic_search.py` | Task4 (dùng chung `embed_texts()`) |
| **TV2** | Task 6: BM25 lexical search, cùng corpus chunks | `task6_lexical_search.py` | Task4 |
| **TV2** | Task 7 (tùy chọn): reranking | `task7_reranking.py` | Task5 + Task6 — bỏ qua nếu hết thời gian |
| **TV3** | Tiếp tục hoàn thiện + test Task8 độc lập | `task8_pageindex_vectorless.py` | — |
| **TV1** | Rảnh tay sau Task4 → hỗ trợ code review, hoặc phụ TV2 viết BM25 | — | — |

Task5 và Task6 là **2 file độc lập nhau**, TV2 có thể tự chia thời gian làm tuần tự hoặc nhờ TV1 hỗ trợ 1 trong 2 để chạy song song nội bộ.

---

## Cụm 3 — Sau khi Task5 + Task6 + Task8 xong

| Thành viên | Việc | File | Phụ thuộc |
|---|---|---|---|
| **TV3** | Task 9: RRF fusion (chạy đúng 1 lần) + fallback threshold (dùng cosine gốc của dense, hiệu chỉnh bằng query in/out-domain) | `task9_retrieval_pipeline.py` | Task5, Task6, Task8 |

Đây là điểm hội tụ (merge point) của cả pipeline retrieval — chỉ 1 người làm để tránh xung đột logic RRF.

---

## Cụm 4 — Sau khi Task9 xong

| Thành viên | Việc | File | Phụ thuộc |
|---|---|---|---|
| **TV4** | Task 10: generation có citation, dispatch theo `LLM_PROVIDER` | `task10_generation.py` | Task9 |
| **TV4** | Nối UI thật (thay mock) vào `retrieve()` + `generate_with_citation()` | `app.py` | Task10 |

---

## Cụm 5 — Cuối cùng, cả nhóm cùng làm

| Việc | Phụ trách | File |
|---|---|---|
| Chạy evaluation thật: 4 metric (faithfulness, answer relevance, context recall, context precision), so sánh dense-only vs hybrid+RRF | TV4 chủ trì, cả nhóm cung cấp câu hỏi domain mình phụ trách | `group_project/evaluation/RESULT.md` |
| `pytest -q` toàn bộ, không còn `NotImplementedError` | Cả nhóm | — |
| Individual report (mỗi người 1 file riêng) | Mỗi người tự làm | `group_project/ịndividual/INDIVIDUAL_REPORT.md` (copy thành file riêng theo tên) |
| Demo: 1 query đúng domain, 1 query ngoài domain, kết quả A/B | Cả nhóm | — |

---

## Sơ đồ phụ thuộc (tóm tắt)

```
TV1: Task1-3 ✅ → Task4 ──┬──────────────┐
                           │              │
TV3: Task8 (song song, độc lập) ──┐       │
                                   │       │
TV4: Golden dataset + UI mock     │       │
     (song song, độc lập)         │       │
                                   ▼       ▼
                    TV2: Task5 + Task6 (cần Task4)
                                   │
                                   ▼
                    TV3: Task9 (cần Task5+6+8)
                                   │
                                   ▼
                    TV4: Task10 + UI thật (cần Task9)
                                   │
                                   ▼
                    Cả nhóm: Evaluation + Test + Demo
```

**Điểm mấu chốt:** TV1 nên hoàn thành Task4 sớm nhất có thể vì đó là điểm khóa duy nhất chặn TV2. TV3 và TV4 có việc độc lập để làm ngay từ ngày đầu, không cần chờ.
