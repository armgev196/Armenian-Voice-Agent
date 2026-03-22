import argparse
import asyncio
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def cmd_scrape() -> None:
    from scraper.run_all import main
    asyncio.run(main())


def cmd_ingest(clear: bool = False) -> None:
    from knowledge_base.ingest import main
    main(clear=clear)


def cmd_agent() -> None:
    from agent.voice_agent import run_worker
    run_worker()


def cmd_setup() -> None:
    logging.info("Step 1/2 — scraping bank websites")
    cmd_scrape()
    logging.info("Step 2/2 — ingesting into vector store")
    cmd_ingest(clear=True)
    logging.info("Setup complete. Start the agent with: python main.py agent")


def cmd_status() -> None:
    from collections import Counter
    from config import DATA_DIR
    from knowledge_base.vectorstore import ArmenianBankVectorStore

    store = ArmenianBankVectorStore()
    stats = store.stats()
    print(f"Vector store: {stats['total_chunks']} chunks ({stats['collection']})")

    raw = DATA_DIR / "scraped_data.json"
    if raw.exists():
        with open(raw) as f:
            docs = json.load(f)
        print(f"Raw documents: {len(docs)}")
        for label, counter in [
            ("bank", Counter(d["bank"] for d in docs)),
            ("topic", Counter(d["topic"] for d in docs)),
        ]:
            for k, v in counter.most_common():
                print(f"  {label}={k}: {v}")
    else:
        print("No scraped data found — run: python main.py scrape")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="main.py")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("scrape", help="Scrape all bank websites")

    ingest = sub.add_parser("ingest", help="Ingest scraped data into ChromaDB")
    ingest.add_argument("--clear", action="store_true", help="Drop existing data first")

    sub.add_parser("agent", help="Start the LiveKit voice agent worker")
    sub.add_parser("setup", help="Scrape + ingest in sequence (first-time setup)")
    sub.add_parser("status", help="Print knowledge base statistics")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    dispatch = {
        "scrape": lambda: cmd_scrape(),
        "ingest": lambda: cmd_ingest(clear=args.clear if hasattr(args, "clear") else False),
        "agent": lambda: cmd_agent(),
        "setup": lambda: cmd_setup(),
        "status": lambda: cmd_status(),
    }

    dispatch[args.command]()


if __name__ == "__main__":
    main()
