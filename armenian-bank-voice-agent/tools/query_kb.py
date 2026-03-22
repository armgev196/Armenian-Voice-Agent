"""
tools/query_kb.py — Interactive CLI to test RAG retrieval without running the full voice agent.

Useful for verifying that the knowledge base has been populated correctly
and that queries return sensible results before testing with voice.

Usage:
    python tools/query_kb.py
    python tools/query_kb.py --bank ameriabank --topic credits
    python tools/query_kb.py --query "What is the mortgage interest rate?"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge_base.vectorstore import ArmenianBankVectorStore
from config import RAG_TOP_K


def interactive_mode(store: ArmenianBankVectorStore, bank=None, topic=None):
    print("\n🇦🇲  Armenian Bank Knowledge Base — Interactive Query")
    print("   Type a question (Armenian or English). Type 'quit' to exit.\n")
    print(f"   Filters: bank={bank or 'all'}, topic={topic or 'all'}")
    print(f"   Retrieving top {RAG_TOP_K} results\n")

    while True:
        try:
            query = input("Query > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        results = store.query(query, top_k=RAG_TOP_K, bank_filter=bank, topic_filter=topic)

        if not results:
            print("  ⚠️  No results found above minimum score threshold.\n")
            continue

        print(f"\n  Found {len(results)} result(s):\n")
        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            print(f"  ─── [{i}] {meta['bank_name']} · {meta['topic']} · score: {r['score']:.3f} ───")
            print(f"  Title: {meta['title']}")
            print(f"  URL:   {meta['url']}")
            preview = r["text"][:300].replace("\n", " ")
            print(f"  Text:  {preview}{'...' if len(r['text']) > 300 else ''}")
            print()


def single_query(store: ArmenianBankVectorStore, query: str, bank=None, topic=None):
    results = store.query(query, top_k=RAG_TOP_K, bank_filter=bank, topic_filter=topic)
    print(f"\nQuery: {query}")
    print(f"Results: {len(results)}\n")

    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        print(f"[{i}] {meta['bank_name']} | {meta['topic']} | score={r['score']:.3f}")
        print(f"     {meta['title']}")
        print(f"     {r['text'][:200]}...\n")


def main():
    parser = argparse.ArgumentParser(description="Query the Armenian Bank knowledge base")
    parser.add_argument("--bank",  help="Filter by bank ID (ameriabank, ardshinbank, acba)")
    parser.add_argument("--topic", help="Filter by topic (credits, deposits, branch_locations)")
    parser.add_argument("--query", help="Single query (non-interactive mode)")
    args = parser.parse_args()

    store = ArmenianBankVectorStore()
    count = store.count()

    if count == 0:
        print("❌ Vector store is empty! Run: python main.py setup")
        sys.exit(1)

    print(f"✅ Knowledge base loaded: {count} chunks")

    if args.query:
        single_query(store, args.query, bank=args.bank, topic=args.topic)
    else:
        interactive_mode(store, bank=args.bank, topic=args.topic)


if __name__ == "__main__":
    main()
