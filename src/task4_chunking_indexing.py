"""
Task 4 — Chunking, embedding và indexing.

Hướng dẫn:
    1. Đọc toàn bộ Markdown trong data/standardized/.
    2. Chia văn bản bằng strategy đã chọn.
    3. Embed chunks bằng một provider duy nhất.
    4. Upsert vào ChromaDB với cosine distance.

Mỗi document/chunk phải theo docs/MODULE_CONTRACTS.md. ID cần ổn định để
chạy lại pipeline không tạo dữ liệu trùng. Task 5 phải dùng chung embed_texts().
"""

import os
import re
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Giải thích lựa chọn tham số trong báo cáo nhóm.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = 1024

COLLECTION_NAME = "rag_documents"

TITLE_PATTERN = re.compile(r"^#\s+(.+)$", re.MULTILINE)
SOURCE_PATTERN = re.compile(r"^\*\*Source:\*\*\s+(\S+)$", re.MULTILINE)


@lru_cache(maxsize=1)
def _sentence_transformer():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    # bge-m3 mặc định max_seq_length=8192; chunk chỉ 500 ký tự nên để nguyên
    # sẽ cấp phát thừa bộ nhớ và làm tiến trình segfault trên Windows.
    model.max_seq_length = 512
    return model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed danh sách text bằng provider được chọn trong .env."""
    if EMBEDDING_PROVIDER == "sentence_transformers":
        return (
            _sentence_transformer()
            .encode(texts, batch_size=8, show_progress_bar=len(texts) > 100)
            .tolist()
        )

    if EMBEDDING_PROVIDER == "openai":
        from openai import OpenAI

        response = OpenAI().embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return [item.embedding for item in response.data]

    if EMBEDDING_PROVIDER == "gemini":
        from google import genai

        response = genai.Client().models.embed_content(
            model=EMBEDDING_MODEL, contents=texts
        )
        return [embedding.values for embedding in response.embeddings]

    raise ValueError(f"EMBEDDING_PROVIDER không hỗ trợ: {EMBEDDING_PROVIDER}")


def get_collection():
    """Mở Chroma collection dùng cosine distance."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def load_documents() -> list[dict]:
    """Đọc Markdown và trả về danh sách Document."""
    documents = []
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = path.read_text(encoding="utf-8")
        if not content.strip():
            continue

        title_match = TITLE_PATTERN.search(content)
        source_match = SOURCE_PATTERN.search(content)
        documents.append(
            {
                "id": path.relative_to(STANDARDIZED_DIR).as_posix(),
                "content": content,
                "metadata": {
                    "source": path.name,
                    "title": title_match.group(1) if title_match else path.stem,
                    "doc_type": "legal" if "legal" in path.parts else "news",
                    "url": source_match.group(1) if source_match else None,
                },
            }
        )
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id và chunk_index."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for document in documents:
        chunk_index = 0
        for text in splitter.split_text(document["content"]):
            text = text.strip()
            if not text:
                continue
            chunks.append(
                {
                    "id": f"{document['id']}::chunk-{chunk_index}",
                    "content": text,
                    "metadata": {
                        **document["metadata"],
                        "chunk_index": chunk_index,
                    },
                }
            )
            chunk_index += 1
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Thêm embedding vào từng chunk."""
    vectors = embed_texts([chunk["content"] for chunk in chunks])
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
    return chunks


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB."""
    collection = get_collection()
    collection.upsert(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["content"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        # ChromaDB từ chối metadata value là None nên url rỗng lưu thành "".
        metadatas=[
            {key: ("" if value is None else value)
             for key, value in chunk["metadata"].items()}
            for chunk in chunks
        ],
    )


def run_pipeline() -> None:
    """Chạy load, chunk, embed và index."""
    documents = load_documents()
    chunks = chunk_documents(documents)
    embedded_chunks = embed_chunks(chunks)
    index_to_vectorstore(embedded_chunks)
    print(f"Indexed {len(embedded_chunks)} chunks")


if __name__ == "__main__":
    run_pipeline()
