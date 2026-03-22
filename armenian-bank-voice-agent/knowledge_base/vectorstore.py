"""
knowledge_base/vectorstore.py — ChromaDB vector store with multilingual embeddings.

Why ChromaDB?
  - Runs fully locally, no external server needed
  - Persistent storage on disk
  - Simple Python API, no infrastructure overhead
  - Scales to millions of documents for future bank additions

Why paraphrase-multilingual-mpnet-base-v2?
  - Explicitly trained on 50+ languages including Armenian
  - 768-dim embeddings, good quality/speed balance
  - Runs on CPU (no GPU required for this workload)
"""

from __future__ import annotations
import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config import CHROMA_DIR, EMBEDDING_MODEL, RAG_TOP_K, RAG_MIN_SCORE

logger = logging.getLogger(__name__)

COLLECTION_NAME = "armenian_bank_knowledge"


class ArmenianBankVectorStore:
    """Manages the ChromaDB collection for Armenian bank knowledge."""

    def __init__(self):
        self._embedding_model: SentenceTransformer | None = None
        self._client: chromadb.PersistentClient | None = None
        self._collection = None

    def _get_embedding_model(self) -> SentenceTransformer:
        if self._embedding_model is None:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            self._embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        return self._embedding_model

    def _get_client(self) -> chromadb.PersistentClient:
        if self._client is None:
            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(CHROMA_DIR),
                settings=Settings(anonymized_telemetry=False),
            )
        return self._client

    def _get_collection(self):
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    # ── Ingest ─────────────────────────────────────────────────────────────────

    def add_documents(self, documents: list[dict]) -> int:
        """
        Add scraped documents to the vector store.
        Chunks long documents before embedding.
        Returns number of chunks added.
        """
        model = self._get_embedding_model()
        collection = self._get_collection()

        chunks = self._chunk_documents(documents)
        if not chunks:
            logger.warning("No chunks to add.")
            return 0

        texts     = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        ids       = [c["id"] for c in chunks]

        # Batch embed (sentence-transformers handles batching internally)
        logger.info(f"Embedding {len(texts)} chunks...")
        embeddings = model.encode(texts, show_progress_bar=True).tolist()

        # Upsert in batches of 500 (ChromaDB limit)
        batch_size = 500
        for i in range(0, len(chunks), batch_size):
            collection.upsert(
                ids=ids[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size],
                documents=texts[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
            )

        logger.info(f"✅ Added {len(chunks)} chunks to vector store.")
        return len(chunks)

    def _chunk_documents(self, documents: list[dict]) -> list[dict]:
        """Split documents into overlapping chunks for better retrieval."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=80,
            separators=["\n\n", "\n", "։", ".", " ", ""],
        )

        chunks: list[dict] = []
        for doc in documents:
            text_chunks = splitter.split_text(doc["content"])
            for idx, chunk in enumerate(text_chunks):
                chunk_id = f"{doc['bank']}_{doc['topic']}_{abs(hash(doc['url']))}_{idx}"
                chunks.append({
                    "id": chunk_id,
                    "text": f"[{doc['bank_name']} — {doc['title']}]\n{chunk}",
                    "metadata": {
                        "bank":      doc["bank"],
                        "bank_name": doc["bank_name"],
                        "topic":     doc["topic"],
                        "title":     doc["title"],
                        "url":       doc["url"],
                        "chunk_idx": idx,
                    },
                })
        return chunks

    # ── Query ──────────────────────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        top_k: int = RAG_TOP_K,
        bank_filter: str | None = None,
        topic_filter: str | None = None,
    ) -> list[dict]:
        """
        Semantic search over the knowledge base.
        Returns list of {text, metadata, score} dicts.
        """
        model      = self._get_embedding_model()
        collection = self._get_collection()

        # Build optional ChromaDB where-filter
        where: dict | None = None
        filters = {}
        if bank_filter:
            filters["bank"] = bank_filter
        if topic_filter:
            filters["topic"] = topic_filter
        if len(filters) == 1:
            where = filters
        elif len(filters) > 1:
            where = {"$and": [{k: v} for k, v in filters.items()]}

        query_embedding = model.encode(query_text).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, max(1, collection.count())),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = 1.0 - dist   # cosine distance → similarity
            if score >= RAG_MIN_SCORE:
                output.append({"text": text, "metadata": meta, "score": score})

        return output

    # ── Utils ──────────────────────────────────────────────────────────────────

    def count(self) -> int:
        return self._get_collection().count()

    def clear(self):
        """Delete and recreate the collection (use for re-ingestion)."""
        client = self._get_client()
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        self._collection = None
        logger.info("Vector store cleared.")

    def stats(self) -> dict:
        collection = self._get_collection()
        total = collection.count()
        return {"total_chunks": total, "collection": COLLECTION_NAME}
