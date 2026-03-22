"""
scraper/acba_scraper.py — Scraper for ACBA Bank (acba.am).
"""

from __future__ import annotations
import logging
from .base_scraper import BaseBankScraper, ScrapedDocument

logger = logging.getLogger(__name__)


class ACBAScraper(BaseBankScraper):
    bank_id = "acba"
    bank_name = "ACBA Bank"
    base_url = "https://www.acba.am"

    async def scrape_credits(self) -> list[ScrapedDocument]:
        docs: list[ScrapedDocument] = []
        urls = [
            f"{self.base_url}/en/individuals/loans",
            f"{self.base_url}/en/individuals/loans/consumer-loans",
            f"{self.base_url}/en/individuals/loans/mortgage",
            f"{self.base_url}/en/individuals/loans/car-loans",
            f"{self.base_url}/en/individuals/credit-cards",
        ]

        for url in urls:
            html = await self._fetch_html(url)
            if not html:
                continue
            soup = self._soup(html)

            for tag in soup.select("header, footer, nav, .menu, script, style"):
                tag.decompose()

            title_el = soup.select_one("h1, .product-name, .page-header h2")
            title = self._clean_text(title_el.get_text()) if title_el else "ACBA Credit"

            content_parts: list[str] = []

            # ACBA uses accordion/tab panels for loan conditions
            for panel in soup.select(".accordion-body, .tab-pane, .product-details, .loan-conditions"):
                t = self._clean_text(panel.get_text())
                if len(t) > 50:
                    content_parts.append(t)

            for p in soup.select("main p, .content-area p, article p"):
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
            f"{self.base_url}/en/individuals/deposits/term-deposits",
            f"{self.base_url}/en/individuals/deposits/savings",
        ]

        for url in urls:
            html = await self._fetch_html(url)
            if not html:
                continue
            soup = self._soup(html)

            for tag in soup.select("header, footer, nav, script, style"):
                tag.decompose()

            title_el = soup.select_one("h1, h2.section-title")
            title = self._clean_text(title_el.get_text()) if title_el else "ACBA Deposit"

            content_parts: list[str] = []
            for block in soup.select(".accordion-body, .tab-pane, main p, .deposit-info"):
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
        url = f"{self.base_url}/en/about-bank/branches-and-atms"
        html = await self._fetch_html(url)
        if not html:
            url = f"{self.base_url}/en/branches"
            html = await self._fetch_html(url)
        if not html:
            return []

        soup = self._soup(html)
        for tag in soup.select("header, footer, nav, script, style"):
            tag.decompose()

        branches: list[str] = []

        for item in soup.select(".branch-item, .branch-info, .location-card"):
            parts = []
            name = item.select_one("h3, h4, .name, strong")
            addr = item.select_one(".address, p")
            phone = item.select_one(".phone, a[href^='tel:']")
            hours = item.select_one(".hours, .schedule")

            if name:
                parts.append(f"Branch: {self._clean_text(name.get_text())}")
            if addr:
                parts.append(f"Address: {self._clean_text(addr.get_text())}")
            if phone:
                parts.append(f"Phone: {self._clean_text(phone.get_text())}")
            if hours:
                parts.append(f"Hours: {self._clean_text(hours.get_text())}")
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
