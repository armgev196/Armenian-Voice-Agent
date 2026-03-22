"""
scraper/run_all.py — Run all bank scrapers and save raw data to JSON.

Usage:
    python -m scraper.run_all
"""

import asyncio
import json
import logging
from pathlib import Path
from tqdm import tqdm
from config import BANKS, SCRAPE_HEADLESS, SCRAPE_TIMEOUT_MS, DATA_DIR
from scraper import AmeriabankScraper, ArdshinbankScraper, ACBAScraper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SCRAPERS = [
    AmeriabankScraper,
    ArdshinbankScraper,
    ACBAScraper,
]


async def run_scraper(scraper_cls) -> list[dict]:
    """Run a single scraper and return list of document dicts."""
    async with scraper_cls(headless=SCRAPE_HEADLESS, timeout_ms=SCRAPE_TIMEOUT_MS) as scraper:
        docs = await scraper.scrape_all()
        return [d.to_dict() for d in docs]


async def main():
    all_docs: list[dict] = []
    output_path = DATA_DIR / "scraped_data.json"

    for scraper_cls in tqdm(SCRAPERS, desc="Scraping banks"):
        logger.info(f"Running {scraper_cls.__name__}...")
        try:
            docs = await run_scraper(scraper_cls)
            all_docs.extend(docs)
            logger.info(f"  → {len(docs)} documents collected")
        except Exception as e:
            logger.error(f"  ✗ Failed: {e}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, ensure_ascii=False, indent=2)

    logger.info(f"\n✅ Scraping complete. {len(all_docs)} total documents saved to {output_path}")

    # Print summary
    from collections import Counter
    by_bank  = Counter(d["bank"]  for d in all_docs)
    by_topic = Counter(d["topic"] for d in all_docs)
    print("\nDocuments by bank:")
    for k, v in by_bank.items():
        print(f"  {k}: {v}")
    print("\nDocuments by topic:")
    for k, v in by_topic.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    asyncio.run(main())
