from __future__ import annotations

import logging

from .base_scraper import BaseBankScraper, ScrapedDocument

logger = logging.getLogger(__name__)

_CREDIT_URLS = [
    "/en/individuals/loans",
    "/en/individuals/loans/consumer",
    "/en/individuals/loans/mortgage",
    "/en/individuals/loans/auto",
    "/en/individuals/credit-cards",
]

_DEPOSIT_URLS = [
    "/en/individuals/deposits",
    "/en/individuals/deposits/term",
    "/en/individuals/deposits/savings",
    "/en/individuals/deposits/demand",
]


class ArdshinbankScraper(BaseBankScraper):
    bank_id = "ardshinbank"
    bank_name = "Ardshinbank"
    base_url = "https://ardshinbank.am"

    async def _scrape_product_pages(self, paths: list[str], topic: str) -> list[ScrapedDocument]:
        docs: list[ScrapedDocument] = []

        for path in paths:
            url = self.base_url + path
            html = await self._fetch_html(url)
            if not html:
                continue

            soup = self._soup(html)
            for tag in soup.select("header, footer, nav, script, style"):
                tag.decompose()

            title_el = soup.select_one("h1, h2.page-title")
            title = self._clean_text(title_el.get_text()) if title_el else f"Ardshinbank {topic.title()}"

            parts: list[str] = []
            for card in soup.select(".product-card, .loan-product, .card, article"):
                card_title = card.select_one("h3, h4, .card-title")
                card_body = card.select_one("p, .card-body, .description")
                segments = []
                if card_title:
                    segments.append(self._clean_text(card_title.get_text()))
                if card_body:
                    segments.append(self._clean_text(card_body.get_text()))
                if segments:
                    parts.append(" — ".join(segments))

            for el in soup.select("main p, .content p, section p"):
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
        for path in ["/en/about/branches", "/en/branches"]:
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
        for item in soup.select(".branch, .branch-item, .location, tr"):
            text = self._clean_text(item.get_text())
            if len(text) > 20:
                branches.append(text)

        if not branches:
            main = soup.select_one("main, #main-content")
            if main:
                branches.append(self._clean_text(main.get_text()))

        if not branches:
            return []

        return [self._make_doc(
            title="Ardshinbank Branches and ATMs",
            content="\n\n".join(branches),
            url=url,
            topic="branch_locations",
        )]
