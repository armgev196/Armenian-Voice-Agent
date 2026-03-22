"""
main.py — CLI entry point for the Armenian Bank Voice Agent.

Commands:
    python main.py scrape      — Scrape all bank websites
    python main.py ingest      — Ingest scraped data into ChromaDB
    python main.py ingest --clear  — Re-ingest from scratch
    python main.py agent       — Start the LiveKit voice agent worker
    python main.py setup       — Run scrape + ingest in sequence (first-time setup)
    python main.py status      — Show knowledge base stats
"""

import argparse
import asyncio
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def cmd_scrape():
    """Run all bank scrapers."""
    from scraper.run_all import main as scrape_main
    asyncio.run(scrape_main())


def cmd_ingest(clear: bool = False):
    """Ingest scraped data into vector store."""
    from knowledge_base.ingest import main as ingest_main
    ingest_main(clear=clear)


def cmd_agent():
    """Start the LiveKit voice agent worker."""
    from agent.voice_agent import run_worker
    run_worker()


def cmd_setup():
    """First-time setup: scrape then ingest."""
    logger.info("=== Step 1/2: Scraping bank websites ===")
    cmd_scrape()
    logger.info("\n=== Step 2/2: Ingesting into vector store ===")
    cmd_ingest(clear=True)
    logger.info("\n✅ Setup complete! Run: python main.py agent")


def cmd_status():
    """Show knowledge base statistics."""
    from knowledge_base.vectorstore import ArmenianBankVectorStore
    import json
    from config import DATA_DIR

    store = ArmenianBankVectorStore()
    stats = store.stats()
    print(f"\n📊 Knowledge Base Status")
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Collection:   {stats['collection']}")

    raw_path = DATA_DIR / "scraped_data.json"
    if raw_path.exists():
        with open(raw_path) as f:
            docs = json.load(f)
        from collections import Counter
        by_bank  = Counter(d["bank"]  for d in docs)
        by_topic = Counter(d["topic"] for d in docs)
        print(f"\n   Raw documents: {len(docs)}")
        print("   By bank:")
        for k, v in by_bank.items():
            print(f"     {k}: {v}")
        print("   By topic:")
        for k, v in by_topic.items():
            print(f"     {k}: {v}")
    else:
        print("   ⚠️  No scraped data found. Run: python main.py scrape")


def main():
    parser = argparse.ArgumentParser(
        description="Armenian Bank Voice AI Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("scrape",  help="Scrape all Armenian bank websites")

    ingest_p = subparsers.add_parser("ingest", help="Ingest scraped data into ChromaDB")
    ingest_p.add_argument("--clear", action="store_true", help="Clear existing data first")

    subparsers.add_parser("agent",   help="Start the LiveKit voice agent")
    subparsers.add_parser("setup",   help="First-time setup (scrape + ingest)")
    subparsers.add_parser("status",  help="Show knowledge base stats")

    args = parser.parse_args()

    if args.command == "scrape":
        cmd_scrape()
    elif args.command == "ingest":
        cmd_ingest(clear=getattr(args, "clear", False))
    elif args.command == "agent":
        cmd_agent()
    elif args.command == "setup":
        cmd_setup()
    elif args.command == "status":
        cmd_status()


if __name__ == "__main__":
    main()
