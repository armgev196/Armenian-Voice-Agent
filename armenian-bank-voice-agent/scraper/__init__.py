"""scraper/__init__.py"""
from .base_scraper import BaseBankScraper, ScrapedDocument
from .ameriabank_scraper import AmeriabankScraper
from .ardshinbank_scraper import ArdshinbankScraper
from .acba_scraper import ACBAScraper

__all__ = [
    "BaseBankScraper",
    "ScrapedDocument",
    "AmeriabankScraper",
    "ArdshinbankScraper",
    "ACBAScraper",
]
