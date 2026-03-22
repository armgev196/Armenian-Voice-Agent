"""
knowledge_base/ingest.py — Load scraped JSON data into the ChromaDB vector store.

Usage:
    python -m knowledge_base.ingest [--clear]
"""

import argparse
import json
import logging
import sys
from config import DATA_DIR
from knowledge_base.vectorstore import ArmenianBankVectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_scraped_data() -> list[dict]:
    path = DATA_DIR / "scraped_data.json"
    if not path.exists():
        logger.error(f"Scraped data not found at {path}. Run: python -m scraper.run_all")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} raw documents from {path}")
    return data


def main(clear: bool = False):
    store = ArmenianBankVectorStore()

    if clear:
        logger.info("Clearing existing vector store...")
        store.clear()

    docs = load_scraped_data()

    # Filter out empty documents
    docs = [d for d in docs if d.get("content") and len(d["content"].strip()) > 50]
    logger.info(f"After filtering: {len(docs)} valid documents")

    n = store.add_documents(docs)
    logger.info(f"\n✅ Ingestion complete. {n} chunks in vector store.")
    logger.info(f"Stats: {store.stats()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest scraped bank data into ChromaDB")
    parser.add_argument("--clear", action="store_true", help="Clear existing data before ingesting")
    args = parser.parse_args()
    main(clear=args.clear)
