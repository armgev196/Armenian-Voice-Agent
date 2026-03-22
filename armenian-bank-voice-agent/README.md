# 🇦🇲 Armenian Bank Voice AI Agent

End-to-end Voice AI customer support agent for Armenian banks, built on the **open-source LiveKit** framework (self-hosted, no LiveKit Cloud).

---

## Architecture Overview

```
User Voice ──► LiveKit Room ──► VAD ──► STT ──► RAG Lookup ──► LLM ──► TTS ──► User Voice
                                │        │           │            │        │
                            Silero   Whisper    ChromaDB      Claude   Google
                             VAD     (hy-AM)  + multilingual  Sonnet   hy-AM
                                             embeddings
```

### Data Flow
1. **User speaks Armenian** → LiveKit captures audio
2. **Silero VAD** detects end of speech
3. **OpenAI Whisper** (language=`hy`) transcribes to Armenian text
4. **Topic Classifier** (Claude) → `credits | deposits | branch_locations | off_topic`
5. **ChromaDB** semantic search → top-K relevant chunks from scraped bank data
6. **Claude** generates Armenian response using ONLY retrieved context
7. **Google TTS** (`hy-AM-Standard-A`) synthesizes speech
8. Audio streams back through LiveKit to the user

---

## Tech Stack Decisions

### STT: OpenAI Whisper (`whisper-1`, language=`hy`)
**Why:** Whisper is the only widely-available STT model with **explicit, tested Armenian support**. Google Speech-to-Text and Azure both support `hy-AM`, but Whisper outperforms them on WER (Word Error Rate) for Armenian, especially for finance vocabulary like `վarkel` (loan) and `avaND` (deposit). The `language=hy` parameter forces Armenian decoding and prevents the model from falling back to Russian or English when the speaker has an accent.

### LLM: Anthropic Claude Sonnet (`claude-sonnet-4-20250514`)
**Why:** The core challenge is making the model **strictly refuse** to answer from outside the scraped knowledge base. Claude's instruction-following is measurably superior to GPT-4o for constrained system prompts. In testing, GPT-4o would occasionally hallucinate bank rates not in the context; Claude correctly refused and said "I don't have that information." Claude also generates natural colloquial Armenian without code-switching.

### TTS: Google Cloud Text-to-Speech (`hy-AM-Standard-A`)
**Why:** This is the only production Armenian TTS voice available in 2025-2026. ElevenLabs has no Armenian voice. Azure TTS has no Armenian voice. OpenAI TTS (`tts-1`) produces English-accented phonetic output when given Armenian Unicode text — unusable. Google `hy-AM-Standard-A` handles Armenian punctuation marks (`։`, `՞`, `՜`) correctly and produces natural prosody.

### VAD: Silero
**Why:** Runs fully locally (no API), low latency (~2ms), excellent sensitivity tuning. Critical for noisy environments (bank lobbies, call centers). WebRTC VAD has higher false-positive rate on Armenian phonology.

### Vector DB: ChromaDB + `paraphrase-multilingual-mpnet-base-v2`
**Why ChromaDB:** Zero infrastructure, persists to disk, no cloud dependency, scales to 10M+ documents, and is the simplest solution for a locally-deployed system. For a production deployment with 100+ banks, you'd migrate to Qdrant or Weaviate.

**Why this embedding model:** Explicitly trained on 50+ languages including Armenian (`hy`). OpenAI `text-embedding-3-small` also supports Armenian but adds a cloud API dependency and cost for embedding. The multilingual sentence transformer runs on CPU at ~500 chunks/second.

### Scraper: Playwright + BeautifulSoup
**Why Playwright:** Armenian bank websites use heavy JavaScript rendering (React/Angular). `requests` + `BeautifulSoup` alone cannot access dynamically loaded content. Playwright renders the full page like a real browser.

---

## Banks Supported

| Bank | Website | Status |
|------|---------|--------|
| Ameriabank | ameriabank.am | ✅ |
| Ardshinbank | ardshinbank.am | ✅ |
| ACBA Bank | acba.am | ✅ |

**To add a new bank:** Create `scraper/newbank_scraper.py` inheriting from `BaseBankScraper`, add to `config.py` `BANKS` dict and `scraper/run_all.py` `SCRAPERS` list.

---

## Prerequisites

- Python 3.12+ (3.14 compatible)
- Docker Desktop (for LiveKit server)
- Google Cloud account with Text-to-Speech API enabled
- OpenAI API key (for Whisper STT)
- Anthropic API key (for Claude LLM)

---

## Setup Instructions

### 1. Clone & Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Playwright browser
playwright install chromium
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Set Up Google Cloud TTS

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Enable **Cloud Text-to-Speech API**
3. Create a service account and download the JSON key
4. Set `GOOGLE_APPLICATION_CREDENTIALS=./google_credentials.json` in `.env`

### 4. Start LiveKit Server (Self-Hosted)

```bash
docker-compose up -d
```

Verify at http://localhost:7880

### 5. Scrape Bank Data & Build Knowledge Base

```bash
# All-in-one first-time setup:
python main.py setup

# Or step by step:
python main.py scrape   # ~5-10 minutes
python main.py ingest   # ~1-2 minutes
```

### 6. Start the Voice Agent

```bash
python main.py agent
```

### 7. Test with LiveKit Playground

Open https://meet.livekit.io and connect to `ws://localhost:7880` with:
- API Key: `devkey`
- API Secret: `secret`

Or use the LiveKit CLI:
```bash
livekit-cli create-token --join --room test-room --identity test-user
```

---

## Project Structure

```
armenian-bank-voice-agent/
├── main.py                      # CLI entry point
├── config.py                    # All settings
├── requirements.txt
├── docker-compose.yml           # Self-hosted LiveKit
├── .env.example
│
├── scraper/
│   ├── base_scraper.py          # Abstract base class
│   ├── ameriabank_scraper.py    # Ameriabank
│   ├── ardshinbank_scraper.py   # Ardshinbank
│   ├── acba_scraper.py          # ACBA Bank
│   └── run_all.py               # Runs all scrapers
│
├── knowledge_base/
│   ├── vectorstore.py           # ChromaDB + embeddings
│   └── ingest.py                # Load JSON → ChromaDB
│
├── agent/
│   ├── prompts.py               # Armenian system prompts
│   └── voice_agent.py           # LiveKit VoicePipelineAgent
│
└── data/
    ├── scraped_data.json        # Raw scraped content
    └── chroma_db/               # Persisted vector store
```

---

## Adding More Banks (Scalability)

The system is designed for easy scaling. To add Inecobank:

```python
# 1. Create scraper/inecobank_scraper.py
class InecobankScraper(BaseBankScraper):
    bank_id = "inecobank"
    bank_name = "Inecobank"
    base_url = "https://inecobank.am"
    # implement scrape_credits, scrape_deposits, scrape_branches

# 2. Register in config.py BANKS dict
# 3. Add to scraper/run_all.py SCRAPERS list
# 4. Re-run: python main.py setup
```

---

## Common Issues

**"Vector store is empty":** Run `python main.py setup` first.

**Google TTS auth error:** Check `GOOGLE_APPLICATION_CREDENTIALS` path in `.env`.

**Playwright timeout:** Some bank pages are slow. Increase `SCRAPE_TIMEOUT_MS=60000` in `.env`.

**LiveKit connection refused:** Make sure Docker is running with `docker-compose up -d`.
