"""
scraper/base_scraper.py — Abstract base class for all Armenian bank scrapers.

Each bank scraper must implement:
  - scrape_credits()   → list of dicts
  - scrape_deposits()  → list of dicts
  - scrape_branches()  → list of dicts

Each dict must have at minimum: {title, content, url, bank, topic}
"""

from __future__ import annotations
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from playwright.async_api import async_playwright, Browser, Page
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)


@dataclass
class ScrapedDocument:
    """A single scraped knowledge chunk."""
    title: str
    content: str
    url: str
    bank: str           # e.g. "ameriabank"
    bank_name: str      # e.g. "Ameriabank"
    topic: str          # "credits" | "deposits" | "branch_locations"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "content": self.content,
            "url": self.url,
            "bank": self.bank,
            "bank_name": self.bank_name,
            "topic": self.topic,
            **self.metadata,
        }


class BaseBankScraper(ABC):
    """Base class for all Armenian bank scrapers."""

    bank_id: str = ""          # e.g. "ameriabank"
    bank_name: str = ""        # e.g. "Ameriabank"
    base_url: str = ""

    def __init__(self, headless: bool = True, timeout_ms: int = 30000):
        self.headless = headless
        self.timeout_ms = timeout_ms
        self._browser: Browser | None = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def __aenter__(self):
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        return self

    async def __aexit__(self, *_):
        if self._browser:
            await self._browser.close()
        await self._playwright.stop()

    # ── Helpers ────────────────────────────────────────────────────────────────

    async def _get_page(self) -> Page:
        """Create a new browser page with default settings."""
        page = await self._browser.new_page()
        page.set_default_timeout(self.timeout_ms)
        await page.set_extra_http_headers({
            "Accept-Language": "hy-AM,hy;q=0.9,en;q=0.8",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
        })
        return page

    async def _fetch_html(self, url: str) -> str:
        """Navigate to URL and return page HTML after JS rendering."""
        page = await self._get_page()
        try:
            await page.goto(url, wait_until="networkidle")
            await page.wait_for_timeout(1500)   # let lazy-loads settle
            return await page.content()
        except Exception as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            return ""
        finally:
            await page.close()

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalize whitespace from scraped text."""
        import re
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _soup(html: str) -> BeautifulSoup:
        return BeautifulSoup(html, "lxml")

    def _make_doc(self, title: str, content: str, url: str,
                  topic: str, **meta) -> ScrapedDocument:
        return ScrapedDocument(
            title=title,
            content=content,
            url=url,
            bank=self.bank_id,
            bank_name=self.bank_name,
            topic=topic,
            metadata=meta,
        )

    # ── Abstract Methods ───────────────────────────────────────────────────────

    @abstractmethod
    async def scrape_credits(self) -> list[ScrapedDocument]:
        """Scrape all credit/loan product pages."""
        ...

    @abstractmethod
    async def scrape_deposits(self) -> list[ScrapedDocument]:
        """Scrape all deposit product pages."""
        ...

    @abstractmethod
    async def scrape_branches(self) -> list[ScrapedDocument]:
        """Scrape branch/ATM location data."""
        ...

    # ── Main Entry ─────────────────────────────────────────────────────────────

    async def scrape_all(self) -> list[ScrapedDocument]:
        """Run all scrapers and return combined documents."""
        logger.info(f"[{self.bank_name}] Starting full scrape...")
        results: list[ScrapedDocument] = []

        for method_name, topic in [
            ("scrape_credits", "credits"),
            ("scrape_deposits", "deposits"),
            ("scrape_branches", "branch_locations"),
        ]:
            try:
                docs = await getattr(self, method_name)()
                logger.info(f"[{self.bank_name}] {topic}: {len(docs)} documents")
                results.extend(docs)
            except Exception as e:
                logger.error(f"[{self.bank_name}] Error in {method_name}: {e}")

        return results
