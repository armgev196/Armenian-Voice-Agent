"""
scraper/ardshinbank_scraper.py — Scraper for Ardshinbank (ardshinbank.am).
"""

from __future__ import annotations
import logging
from .base_scraper import BaseBankScraper, ScrapedDocument

logger = logging.getLogger(__name__)


class ArdshinbankScraper(BaseBankScraper):
    bank_id = "ardshinbank"
    bank_name = "Ardshinbank"
    base_url = "https://ardshinbank.am"

    async def scrape_credits(self) -> list[ScrapedDocument]:
        docs: list[ScrapedDocument] = []
        urls = [
            f"{self.base_url}/en/individuals/loans",
            f"{self.base_url}/en/individuals/loans/consumer",
            f"{self.base_url}/en/individuals/loans/mortgage",
            f"{self.base_url}/en/individuals/loans/auto",
            f"{self.base_url}/en/individuals/credit-cards",
        ]

        for url in urls:
            html = await self._fetch_html(url)
            if not html:
                continue
            soup = self._soup(html)

            for tag in soup.select("header, footer, nav, .header, .footer, script, style"):
                tag.decompose()

            title_el = soup.select_one("h1, h2.page-title")
            title = self._clean_text(title_el.get_text()) if title_el else "Ardshinbank Loan"

            content_parts: list[str] = []

            # Loan product cards
            for card in soup.select(".product-card, .loan-product, .card, article"):
                card_title = card.select_one("h3, h4, .card-title")
                card_body  = card.select_one("p, .card-body, .description")
                parts = []
                if card_title:
                    parts.append(self._clean_text(card_title.get_text()))
                if card_body:
                    parts.append(self._clean_text(card_body.get_text()))
                if parts:
                    content_parts.append(" — ".join(parts))

            for p in soup.select("main p, .content p, section p"):
                t = self._clean_text(p.get_text())
                if len(t) > 50:
                    content_parts.append(t)

            for table in soup.select("table"):
                rows = []
                for tr in table.select("tr"):
                    cells = [self._clean_text(td.get_text()) for td in tr.select("td, th")]
                    if any(c for c in cells if c):
                        rows.append(" | ".join(cells))
                if rows:
                    content_parts.append("\n".join(rows))

            content = "\n\n".join(dict.fromkeys(content_parts))
            if len(content) > 100:
                docs.append(self._make_doc(title=title, content=content, url=url, topic="credits"))

        return docs

    async def scrape_deposits(self) -> list[ScrapedDocument]:
        docs: list[ScrapedDocument] = []
        urls = [
            f"{self.base_url}/en/individuals/deposits",
            f"{self.base_url}/en/individuals/deposits/term",
            f"{self.base_url}/en/individuals/deposits/savings",
            f"{self.base_url}/en/individuals/deposits/demand",
        ]

        for url in urls:
            html = await self._fetch_html(url)
            if not html:
                continue
            soup = self._soup(html)

            for tag in soup.select("header, footer, nav, script, style"):
                tag.decompose()

            title_el = soup.select_one("h1")
            title = self._clean_text(title_el.get_text()) if title_el else "Ardshinbank Deposit"

            content_parts: list[str] = []
            for block in soup.select("main p, .deposit-card, .product-card, .content p"):
                t = self._clean_text(block.get_text())
                if len(t) > 50:
                    content_parts.append(t)

            for table in soup.select("table"):
                rows = []
                for tr in table.select("tr"):
                    cells = [self._clean_text(td.get_text()) for td in tr.select("td, th")]
                    if any(c for c in cells if c):
                        rows.append(" | ".join(cells))
                if rows:
                    content_parts.append("\n".join(rows))

            content = "\n\n".join(dict.fromkeys(content_parts))
            if len(content) > 100:
                docs.append(self._make_doc(title=title, content=content, url=url, topic="deposits"))

        return docs

    async def scrape_branches(self) -> list[ScrapedDocument]:
        url = f"{self.base_url}/en/about/branches"
        html = await self._fetch_html(url)
        if not html:
            # try alternate URL
            url = f"{self.base_url}/en/branches"
            html = await self._fetch_html(url)
        if not html:
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
