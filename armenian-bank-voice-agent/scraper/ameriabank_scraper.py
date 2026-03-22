"""
scraper/ameriabank_scraper.py — Scraper for Ameriabank (ameriabank.am).
"""

from __future__ import annotations
import logging
from .base_scraper import BaseBankScraper, ScrapedDocument

logger = logging.getLogger(__name__)


class AmeriabankScraper(BaseBankScraper):
    bank_id = "ameriabank"
    bank_name = "Ameriabank"
    base_url = "https://ameriabank.am"

    # ── Credits ────────────────────────────────────────────────────────────────

    async def scrape_credits(self) -> list[ScrapedDocument]:
        docs: list[ScrapedDocument] = []
        credit_urls = [
            f"{self.base_url}/en/for-individuals/loans/consumer-loan",
            f"{self.base_url}/en/for-individuals/loans/mortgage",
            f"{self.base_url}/en/for-individuals/loans/car-loan",
            f"{self.base_url}/en/for-individuals/loans/credit-card",
            f"{self.base_url}/en/for-individuals/loans",
        ]

        for url in credit_urls:
            html = await self._fetch_html(url)
            if not html:
                continue
            soup = self._soup(html)

            # Remove nav/footer noise
            for tag in soup.select("header, footer, nav, .cookie-banner, script, style"):
                tag.decompose()

            title_el = soup.select_one("h1, .page-title, .product-title")
            title = self._clean_text(title_el.get_text()) if title_el else "Ameriabank Credit"

            # Grab main content blocks
            content_parts: list[str] = []

            # Product descriptions
            for block in soup.select(".product-description, .loan-info, .content-block, main p, .tab-content"):
                text = self._clean_text(block.get_text())
                if len(text) > 40:
                    content_parts.append(text)

            # Tables (rates, terms)
            for table in soup.select("table"):
                rows = []
                for tr in table.select("tr"):
                    cells = [self._clean_text(td.get_text()) for td in tr.select("td, th")]
                    if any(cells):
                        rows.append(" | ".join(cells))
                if rows:
                    content_parts.append("\n".join(rows))

            content = "\n\n".join(dict.fromkeys(content_parts))   # deduplicate
            if len(content) > 100:
                docs.append(self._make_doc(
                    title=title,
                    content=content,
                    url=url,
                    topic="credits",
                ))

        return docs

    # ── Deposits ───────────────────────────────────────────────────────────────

    async def scrape_deposits(self) -> list[ScrapedDocument]:
        docs: list[ScrapedDocument] = []
        deposit_urls = [
            f"{self.base_url}/en/for-individuals/deposits/term-deposit",
            f"{self.base_url}/en/for-individuals/deposits/savings-account",
            f"{self.base_url}/en/for-individuals/deposits",
        ]

        for url in deposit_urls:
            html = await self._fetch_html(url)
            if not html:
                continue
            soup = self._soup(html)

            for tag in soup.select("header, footer, nav, script, style"):
                tag.decompose()

            title_el = soup.select_one("h1, .page-title")
            title = self._clean_text(title_el.get_text()) if title_el else "Ameriabank Deposit"

            content_parts: list[str] = []
            for block in soup.select("main p, .content-block, .deposit-info, .tab-content"):
                text = self._clean_text(block.get_text())
                if len(text) > 40:
                    content_parts.append(text)

            for table in soup.select("table"):
                rows = []
                for tr in table.select("tr"):
                    cells = [self._clean_text(td.get_text()) for td in tr.select("td, th")]
                    if any(cells):
                        rows.append(" | ".join(cells))
                if rows:
                    content_parts.append("\n".join(rows))

            content = "\n\n".join(dict.fromkeys(content_parts))
            if len(content) > 100:
                docs.append(self._make_doc(
                    title=title,
                    content=content,
                    url=url,
                    topic="deposits",
                ))

        return docs

    # ── Branches ───────────────────────────────────────────────────────────────

    async def scrape_branches(self) -> list[ScrapedDocument]:
        url = f"{self.base_url}/en/about-us/branch-network"
        html = await self._fetch_html(url)
        if not html:
            return []

        soup = self._soup(html)
        for tag in soup.select("header, footer, nav, script, style"):
            tag.decompose()

        branches: list[str] = []

        # Branch cards / list items
        for branch in soup.select(".branch-item, .branch-card, .location-item, li.branch"):
            name_el = branch.select_one("h3, h4, .branch-name, strong")
            addr_el = branch.select_one(".address, .branch-address, p")
            hours_el = branch.select_one(".hours, .working-hours, .schedule")
            phone_el = branch.select_one(".phone, a[href^='tel:']")

            parts = []
            if name_el:
                parts.append(f"Branch: {self._clean_text(name_el.get_text())}")
            if addr_el:
                parts.append(f"Address: {self._clean_text(addr_el.get_text())}")
            if hours_el:
                parts.append(f"Hours: {self._clean_text(hours_el.get_text())}")
            if phone_el:
                parts.append(f"Phone: {self._clean_text(phone_el.get_text())}")
            if parts:
                branches.append("\n".join(parts))

        # Fallback: grab all text if structured elements not found
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
