"""
config.py — Central configuration for the Armenian Bank Voice Agent.
All settings are loaded from environment variables (via .env file).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Project Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
CHROMA_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", str(DATA_DIR / "chroma_db")))

# ── LiveKit ────────────────────────────────────────────────────────────────────
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "secret")

# ── API Keys ───────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── Model Choices (see README for rationale) ───────────────────────────────────
STT_MODEL = "whisper-1"          # OpenAI Whisper — best Armenian (hy) support
STT_LANGUAGE = "hy"              # ISO 639-1 code for Armenian

LLM_MODEL = "claude-sonnet-4-20250514"   # Best instruction-following for strict RAG
LLM_MAX_TOKENS = 512

TTS_LANGUAGE_CODE = "hy-AM"             # Google Cloud TTS Armenian
TTS_VOICE_NAME = "hy-AM-Standard-A"     # Only available Armenian voice on Google
TTS_SPEAKING_RATE = 0.95                # Slightly slower for clarity

# ── RAG Settings ──────────────────────────────────────────────────────────────
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.30"))
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"  # Supports Armenian

# ── Scraper Settings ──────────────────────────────────────────────────────────
SCRAPE_HEADLESS = os.getenv("SCRAPE_HEADLESS", "true").lower() == "true"
SCRAPE_TIMEOUT_MS = int(os.getenv("SCRAPE_TIMEOUT_MS", "30000"))

# ── Supported Banks (add more here to scale) ──────────────────────────────────
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

# Topics the agent is allowed to answer about
ALLOWED_TOPICS = ["credits", "deposits", "branch_locations"]
