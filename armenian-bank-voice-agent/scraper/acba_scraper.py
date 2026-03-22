from __future__ import annotations

import logging

from .base_scraper import BaseBankScraper, ScrapedDocument

logger = logging.getLogger(__name__)

_CREDIT_URLS = [
    "/en/individuals/loans",
    "/en/individuals/loans/consumer-loans",
    "/en/individuals/loans/mortgage",
    "/en/individuals/loans/car-loans",
    "/en/individuals/credit-cards",
]

_DEPOSIT_URLS = [
    "/en/individuals/deposits",
    "/en/individuals/deposits/term-deposits",
    "/en/individuals/deposits/savings",
]


class ACBAScraper(BaseBankScraper):
    bank_id = "acba"
    bank_name = "ACBA Bank"
    base_url = "https://www.acba.am"

    async def _scrape_product_pages(self, paths: list[str], topic: str) -> list[ScrapedDocument]:
        docs: list[ScrapedDocument] = []

        for path in paths:
            url = self.base_url + path
            html = await self._fetch_html(url)
            if not html:
                continue

            soup = self._soup(html)
            for tag in soup.select("header, footer, nav, .menu, script, style"):
                tag.decompose()

            title_el = soup.select_one("h1, .product-name, .page-header h2")
            title = self._clean_text(title_el.get_text()) if title_el else f"ACBA {topic.title()}"

            parts: list[str] = []
            for block in soup.select(".accordion-body, .tab-pane, .product-details, .loan-conditions"):
                text = self._clean_text(block.get_text())
                if len(text) > 50:
                    parts.append(text)

            for el in soup.select("main p, .content-area p, article p"):
                text = self._clean_text(el.get_text())
                if len(text) > 50:
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
        for path in ["/en/about-bank/branches-and-atms", "/en/branches"]:
            url = self.base_url + path
            html = await self._fetch_html(url)
            if html:
                break
        else:
            return []

        soup = self._soup(html)
        for tag in soup.select("header, footer, nav, script, style"):
            tag.decompose()

        branches: list[str] = []
        for item in soup.select(".branch-item, .branch-info, .location-card"):
            parts = []
            for selector, label in [
                ("h3, h4, .name, strong", "Branch"),
                (".address, p", "Address"),
                (".phone, a[href^='tel:']", "Phone"),
                (".hours, .schedule", "Hours"),
            ]:
                el = item.select_one(selector)
                if el:
                    parts.append(f"{label}: {self._clean_text(el.get_text())}")
            if parts:
                branches.append("\n".join(parts))

        if not branches:
            main = soup.select_one("main, #main")
            if main:
                branches.append(self._clean_text(main.get_text()))

        if not branches:
            return []

        return [self._make_doc(
            title="ACBA Bank Branches and ATMs",
            content="\n\n---\n\n".join(branches),
            url=url,
            topic="branch_locations",
        )]
