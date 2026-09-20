"""
Task 4 — Chunking, embedding và indexing.

Hướng dẫn:
    1. Đọc toàn bộ Markdown trong data/standardized/.
    2. Chia văn bản bằng strategy đã chọn (RecursiveCharacterTextSplitter).
    3. Embed chunks bằng một provider duy nhất (sentence_transformers hoặc gemini).
    4. Upsert vào ChromaDB với cosine distance.

Mỗi document/chunk phải theo docs/MODULE_CONTRACTS.md. ID cần ổn định để
chạy lại pipeline không tạo dữ liệu trùng. Task 5 phải dùng chung embed_texts().
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers").lower()
EMBEDDING_DIM = 1024

COLLECTION_NAME = "rag_documents"

_EMBEDDER_MODEL = None


def _get_sentence_transformer():
    global _EMBEDDER_MODEL
    if _EMBEDDER_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER_MODEL = SentenceTransformer(EMBEDDING_MODEL)
    return _EMBEDDER_MODEL


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Tạo vector embeddings cho danh sách text."""
    if not texts:
        return []

    provider = os.getenv("EMBEDDING_PROVIDER", EMBEDDING_PROVIDER).lower()

    if provider in ("gemini", "google"):
        import time
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("gemini_api_key") or os.getenv("GOOGLE_API_KEY")
        model_name = os.getenv("EMBEDDING_MODEL") or "gemini-embedding-001"
        if "gemini" not in model_name:
            model_name = "gemini-embedding-001"

        from google import genai
        client = genai.Client(api_key=api_key)
        all_embeddings = []
        batch_size = 100

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    result = client.models.embed_content(
                        model=model_name,
                        contents=batch,
                    )
                    all_embeddings.extend([e.values for e in result.embeddings])
                    time.sleep(2.0)
                    break
                except Exception as e:
                    err_str = str(e)
                    if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str) and attempt < max_retries - 1:
                        print(f"[Gemini Quota Limit] Dang cho 13s de hoi phục quota (lan {attempt+1}/{max_retries})...")
                        time.sleep(13.0)
                    else:
                        raise e
        return all_embeddings

    # Mặc định sử dụng sentence-transformers cục bộ
    model = _get_sentence_transformer()
    return model.encode(texts).tolist()


def get_collection():
    """Mở hoặc tạo Chroma persistent collection dùng cosine distance."""
    import chromadb
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def load_documents() -> list[dict]:
    """Đọc mọi file Markdown trong data/standardized/ và trả về Document theo contract."""
    documents = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for path in STANDARDIZED_DIR.rglob("*.md"):
        doc_type = "legal" if "legal" in path.parts else "news"
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        rel_id = path.relative_to(STANDARDIZED_DIR).as_posix()
        documents.append({
            "id": rel_id,
            "content": content,
            "metadata": {
                "source": path.name,
                "title": path.stem,
                "doc_type": doc_type,
                "url": None,
            },
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id duy nhất và chunk_index tăng dần."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for document in documents:
        splits = splitter.split_text(document["content"])
        if not splits:
            splits = [document["content"]]
        for index, text in enumerate(splits):
            chunk_id = f"{document['id']}::chunk-{index}"
            chunks.append({
                "id": chunk_id,
                "content": text,
                "metadata": {
                    **document["metadata"],
                    "chunk_index": index,
                },
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Thêm trường embedding vector vào từng chunk."""
    if not chunks:
        return []
    texts = [chunk["content"] for chunk in chunks]
    vectors = embed_texts(texts)
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
    return chunks


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB."""
    if not chunks:
        return
    collection = get_collection()
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        # ChromaDB không chấp nhận value là None trong metadata, đổi thành ""
        clean_metadatas = [
            {k: ("" if v is None else v) for k, v in chunk["metadata"].items()}
            for chunk in batch
        ]
        collection.upsert(
            ids=[chunk["id"] for chunk in batch],
            documents=[chunk["content"] for chunk in batch],
            embeddings=[chunk["embedding"] for chunk in batch],
            metadatas=clean_metadatas,
        )


def run_pipeline() -> None:
    """Chạy load, chunk, embed và index."""
    documents = load_documents()
    print(f"Loaded {len(documents)} documents.")
    chunks = chunk_documents(documents)
    print(f"Generated {len(chunks)} chunks.")
    embedded_chunks = embed_chunks(chunks)
    index_to_vectorstore(embedded_chunks)
    print(f"Successfully indexed {len(embedded_chunks)} chunks to ChromaDB.")


if __name__ == "__main__":
    run_pipeline()
