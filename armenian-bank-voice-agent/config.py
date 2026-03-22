import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

CHROMA_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", str(DATA_DIR / "chroma_db")))

LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "secret")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

STT_MODEL = "whisper-1"
STT_LANGUAGE = "hy"

LLM_MODEL = "claude-sonnet-4-20250514"
LLM_MAX_TOKENS = 512

TTS_LANGUAGE_CODE = "hy-AM"
TTS_VOICE_NAME = "hy-AM-Standard-A"
TTS_SPEAKING_RATE = 0.95

RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.30"))
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"

SCRAPE_HEADLESS = os.getenv("SCRAPE_HEADLESS", "true").lower() == "true"
SCRAPE_TIMEOUT_MS = int(os.getenv("SCRAPE_TIMEOUT_MS", "30000"))

BANKS = {
    "ameriabank": {
        "name": "Ameriabank",
        "name_hy": "Ամերիաբանկ",
        "base_url": "https://ameriabank.am",
        "scraper_class": "scraper.ameriabank_scraper.AmeriabankScraper",
    },
    "ardshinbank": {
        "name": "Ardshinbank",
        "name_hy": "Արդշինբանկ",
        "base_url": "https://ardshinbank.am",
        "scraper_class": "scraper.ardshinbank_scraper.ArdshinbankScraper",
    },
    "acba": {
        "name": "ACBA Bank",
        "name_hy": "ԱԿԲԱ Բանկ",
        "base_url": "https://www.acba.am",
        "scraper_class": "scraper.acba_scraper.ACBAScraper",
    },
}

ALLOWED_TOPICS = {"credits", "deposits", "branch_locations"}
