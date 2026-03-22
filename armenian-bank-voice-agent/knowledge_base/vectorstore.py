from __future__ import annotations

import logging
from typing import Optional

import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from config import CHROMA_DIR, EMBEDDING_MODEL, RAG_MIN_SCORE, RAG_TOP_K

logger = logging.getLogger(__name__)

COLLECTION_NAME = "armenian_bank_knowledge"
UPSERT_BATCH_SIZE = 500


class ArmenianBankVectorStore:
    def __init__(self) -> None:
        self._model: Optional[SentenceTransformer] = None
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
            self._model = SentenceTransformer(EMBEDDING_MODEL)
        return self._model

    @property
    def client(self) -> chromadb.PersistentClient:
        if self._client is None:
            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(CHROMA_DIR),
                settings=Settings(anonymized_telemetry=False),
            )
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add_documents(self, documents: list[dict]) -> int:
        chunks = self._chunk_documents(documents)
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True).tolist()

        for i in range(0, len(chunks), UPSERT_BATCH_SIZE):
            batch = chunks[i : i + UPSERT_BATCH_SIZE]
            self.collection.upsert(
                ids=[c["id"] for c in batch],
                embeddings=embeddings[i : i + UPSERT_BATCH_SIZE],
                documents=[c["text"] for c in batch],
                metadatas=[c["metadata"] for c in batch],
            )

        logger.info("Upserted %d chunks", len(chunks))
        return len(chunks)

    def _chunk_documents(self, documents: list[dict]) -> list[dict]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=80,
            separators=["\n\n", "\n", "։", ".", " ", ""],
        )

        chunks: list[dict] = []
        for doc in documents:
            for idx, text in enumerate(splitter.split_text(doc["content"])):
                chunk_id = f"{doc['bank']}_{doc['topic']}_{abs(hash(doc['url']))}_{idx}"
                chunks.append({
                    "id": chunk_id,
                    "text": f"[{doc['bank_name']} — {doc['title']}]\n{text}",
                    "metadata": {
                        "bank": doc["bank"],
                        "bank_name": doc["bank_name"],
                        "topic": doc["topic"],
                        "title": doc["title"],
                        "url": doc["url"],
                        "chunk_idx": idx,
                    },
                })
        return chunks

    def query(
        self,
        query_text: str,
        top_k: int = RAG_TOP_K,
        bank_filter: Optional[str] = None,
        topic_filter: Optional[str] = None,
    ) -> list[dict]:
        total = self.collection.count()
        if total == 0:
            return []

        where = self._build_filter(bank_filter, topic_filter)
        embedding = self.model.encode(query_text).tolist()

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, total),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        return [
            {"text": text, "metadata": meta, "score": round(1.0 - dist, 4)}
            for text, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
            if 1.0 - dist >= RAG_MIN_SCORE
        ]

    @staticmethod
    def _build_filter(
        bank: Optional[str], topic: Optional[str]
    ) -> Optional[dict]:
        filters = {k: v for k, v in [("bank", bank), ("topic", topic)] if v}
        if not filters:
            return None
        if len(filters) == 1:
            return filters
        return {"$and": [{k: v} for k, v in filters.items()]}

    def count(self) -> int:
        return self.collection.count()

    def clear(self) -> None:
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        self._collection = None
        logger.info("Collection cleared")

    def stats(self) -> dict:
        return {"total_chunks": self.collection.count(), "collection": COLLECTION_NAME}
