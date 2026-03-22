# Armenian Bank Voice Agent

Voice AI agent for Armenian bank customer support. Self-hosted on LiveKit (no cloud dependency). Answers questions about credits, deposits, and branch locations in Armenian.

## How it works

```
mic → LiveKit → Silero VAD → Whisper STT (hy) → topic classifier → ChromaDB similarity search
→ Claude (context-only prompt) → Google TTS (hy-AM) → speaker
```

Each call goes through topic classification first. Off-topic questions are rejected before hitting the RAG pipeline. The LLM system prompt hard-blocks responses outside the retrieved context — the model is not allowed to draw on general knowledge.

## Banks

| Bank | Domain |
|------|--------|
| Ameriabank | ameriabank.am |
| Ardshinbank | ardshinbank.am |
| ACBA Bank | acba.am |

To add another bank, subclass `BaseBankScraper`, implement `scrape_credits`, `scrape_deposits`, and `scrape_branches`, then register it in `config.BANKS` and `scraper/run_all.SCRAPERS`.

## Prerequisites

- Python 3.12+
- Docker (for LiveKit)
- Google Cloud project with Text-to-Speech API enabled
- OpenAI API key
- Anthropic API key

## Setup

```bash
pip install -r requirements.txt
playwright install chromium

cp .env.example .env
# fill in API keys

# Google TTS: create a service account, download the JSON key,
# set GOOGLE_APPLICATION_CREDENTIALS=./google_credentials.json in .env

docker-compose up -d

python main.py setup   # scrape all banks + build vector store (~10 min)
```

## Running

```bash
# In separate terminals:
python token_server.py      # port 8080 — issues LiveKit JWTs to browser clients
python main.py agent        # LiveKit worker process
python tools/serve_ui.py    # opens browser test UI at localhost:3000
```

## Commands

```
python main.py scrape          scrape all bank websites → data/scraped_data.json
python main.py ingest          embed + load into ChromaDB
python main.py ingest --clear  re-ingest from scratch
python main.py agent           start the voice agent worker
python main.py setup           scrape + ingest in sequence
python main.py status          print knowledge base stats
```

## Troubleshooting

**Empty vector store on agent start** — run `python main.py setup` first.

**Google TTS `UNAUTHENTICATED`** — verify the path in `GOOGLE_APPLICATION_CREDENTIALS` and that the service account has the `Cloud Text-to-Speech API User` role.

**Playwright timeout during scrape** — bank sites can be slow; bump `SCRAPE_TIMEOUT_MS=60000` in `.env`.

**LiveKit `connection refused`** — check Docker is running: `docker-compose ps`.

## Tests

```bash
python -m pytest tests/ -v
```
