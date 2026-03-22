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
        logger.error("No scraped data at %s — run: python main.py scrape", path)
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main(clear: bool = False) -> None:
    store = ArmenianBankVectorStore()

    if clear:
        store.clear()

    docs = [d for d in load_scraped_data() if len(d.get("content", "").strip()) > 50]
    logger.info("Ingesting %d documents", len(docs))

    n = store.add_documents(docs)
    logger.info("Done — %d chunks in store", n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()
    main(clear=args.clear)
