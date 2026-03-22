from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from bs4 import BeautifulSoup
from playwright.async_api import Browser, Page, async_playwright

logger = logging.getLogger(__name__)


@dataclass
class ScrapedDocument:
    title: str
    content: str
    url: str
    bank: str
    bank_name: str
    topic: str
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
    bank_id: str = ""
    bank_name: str = ""
    base_url: str = ""

    def __init__(self, headless: bool = True, timeout_ms: int = 30000):
        self.headless = headless
        self.timeout_ms = timeout_ms
        self._browser: Browser | None = None

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

    async def _get_page(self) -> Page:
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
        page = await self._get_page()
        try:
            await page.goto(url, wait_until="networkidle")
            await page.wait_for_timeout(1500)
            return await page.content()
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", url, e)
            return ""
        finally:
            await page.close()

    @staticmethod
    def _clean_text(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _soup(html: str) -> BeautifulSoup:
        return BeautifulSoup(html, "lxml")

    def _extract_tables(self, soup: BeautifulSoup) -> list[str]:
        tables = []
        for table in soup.select("table"):
            rows = []
            for tr in table.select("tr"):
                cells = [self._clean_text(td.get_text()) for td in tr.select("td, th")]
                if any(cells):
                    rows.append(" | ".join(cells))
            if rows:
                tables.append("\n".join(rows))
        return tables

    def _make_doc(
        self, title: str, content: str, url: str, topic: str, **meta
    ) -> ScrapedDocument:
        return ScrapedDocument(
            title=title,
            content=content,
            url=url,
            bank=self.bank_id,
            bank_name=self.bank_name,
            topic=topic,
            metadata=meta,
        )

    @abstractmethod
    async def scrape_credits(self) -> list[ScrapedDocument]: ...

    @abstractmethod
    async def scrape_deposits(self) -> list[ScrapedDocument]: ...

    @abstractmethod
    async def scrape_branches(self) -> list[ScrapedDocument]: ...

    async def scrape_all(self) -> list[ScrapedDocument]:
        logger.info("[%s] Starting scrape", self.bank_name)
        results: list[ScrapedDocument] = []

        scrapers = [
            ("scrape_credits", "credits"),
            ("scrape_deposits", "deposits"),
            ("scrape_branches", "branch_locations"),
        ]

        for method_name, topic in scrapers:
            try:
                docs = await getattr(self, method_name)()
                logger.info("[%s] %s: %d documents", self.bank_name, topic, len(docs))
                results.extend(docs)
            except Exception as e:
                logger.error("[%s] %s failed: %s", self.bank_name, method_name, e)

        return results
