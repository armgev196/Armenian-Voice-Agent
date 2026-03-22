from __future__ import annotations

import logging

from .base_scraper import BaseBankScraper, ScrapedDocument

logger = logging.getLogger(__name__)

_CREDIT_URLS = [
    "/en/for-individuals/loans",
    "/en/for-individuals/loans/consumer-loan",
    "/en/for-individuals/loans/mortgage",
    "/en/for-individuals/loans/car-loan",
    "/en/for-individuals/loans/credit-card",
]

_DEPOSIT_URLS = [
    "/en/for-individuals/deposits",
    "/en/for-individuals/deposits/term-deposit",
    "/en/for-individuals/deposits/savings-account",
]


class AmeriabankScraper(BaseBankScraper):
    bank_id = "ameriabank"
    bank_name = "Ameriabank"
    base_url = "https://ameriabank.am"

    async def _scrape_product_pages(self, paths: list[str], topic: str) -> list[ScrapedDocument]:
        docs: list[ScrapedDocument] = []

        for path in paths:
            url = self.base_url + path
            html = await self._fetch_html(url)
            if not html:
                continue

            soup = self._soup(html)
            for tag in soup.select("header, footer, nav, script, style, .cookie-banner"):
                tag.decompose()

            title_el = soup.select_one("h1, .page-title, .product-title")
            title = self._clean_text(title_el.get_text()) if title_el else f"Ameriabank {topic.title()}"

            parts: list[str] = []
            for block in soup.select(".product-description, .loan-info, .content-block, main p, .tab-content"):
                text = self._clean_text(block.get_text())
                if len(text) > 40:
                    parts.append(text)

            parts.extend(self._extract_tables(soup))

            content = "\n\n".join(dict.fromkeys(parts))
            if len(content) > 100:
                docs.append(self._make_doc(title=title, content=content, url=url, topic=topic))

        return docs

    async def scrape_credits(self) -> list[ScrapedDocument]:
        return await self._scrape_product_pages(_CREDIT_URLS, "credits")

    async def scrape_deposits(self) -> list[ScrapedDocument]:
        return await self._scrape_product_pages(_DEPOSIT_URLS, "deposits")

    async def scrape_branches(self) -> list[ScrapedDocument]:
        url = f"{self.base_url}/en/about-us/branch-network"
        html = await self._fetch_html(url)
        if not html:
            return []

        soup = self._soup(html)
        for tag in soup.select("header, footer, nav, script, style"):
            tag.decompose()

        branches: list[str] = []
        for item in soup.select(".branch-item, .branch-card, .location-item, li.branch"):
            parts = []
            for selector, label in [
                ("h3, h4, .branch-name, strong", "Branch"),
                (".address, .branch-address, p", "Address"),
                (".hours, .working-hours, .schedule", "Hours"),
                (".phone, a[href^='tel:']", "Phone"),
            ]:
                el = item.select_one(selector)
                if el:
                    parts.append(f"{label}: {self._clean_text(el.get_text())}")
            if parts:
                branches.append("\n".join(parts))

        if not branches:
            main = soup.select_one("main, .content, #content")
            if main:
                text = self._clean_text(main.get_text())
                if len(text) > 100:
                    branches.append(text)

        if not branches:
            return []

        return [self._make_doc(
            title="Ameriabank Branch Network",
            content="\n\n---\n\n".join(branches),
            url=url,
            topic="branch_locations",
        )]
