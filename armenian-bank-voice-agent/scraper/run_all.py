import asyncio
import json
import logging
from collections import Counter

from tqdm import tqdm

from config import DATA_DIR, SCRAPE_HEADLESS, SCRAPE_TIMEOUT_MS
from scraper import ACBAScraper, AmeriabankScraper, ArdshinbankScraper
from scraper.base_scraper import BaseBankScraper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SCRAPERS: list[type[BaseBankScraper]] = [
    AmeriabankScraper,
    ArdshinbankScraper,
    ACBAScraper,
]


async def run_scraper(cls: type[BaseBankScraper]) -> list[dict]:
    async with cls(headless=SCRAPE_HEADLESS, timeout_ms=SCRAPE_TIMEOUT_MS) as scraper:
        docs = await scraper.scrape_all()
        return [d.to_dict() for d in docs]


async def main() -> None:
    all_docs: list[dict] = []

    for cls in tqdm(SCRAPERS, desc="Banks"):
        try:
            docs = await run_scraper(cls)
            all_docs.extend(docs)
            logger.info("%s: %d documents", cls.bank_name, len(docs))
        except Exception as e:
            logger.error("%s failed: %s", cls.__name__, e)

    output = DATA_DIR / "scraped_data.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, ensure_ascii=False, indent=2)

    logger.info("Saved %d documents to %s", len(all_docs), output)

    by_bank = Counter(d["bank"] for d in all_docs)
    by_topic = Counter(d["topic"] for d in all_docs)
    for label, counter in [("bank", by_bank), ("topic", by_topic)]:
        for k, v in counter.items():
            logger.info("  %s=%s: %d", label, k, v)


if __name__ == "__main__":
    asyncio.run(main())
