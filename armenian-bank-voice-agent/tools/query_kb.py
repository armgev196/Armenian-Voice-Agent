import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RAG_TOP_K
from knowledge_base.vectorstore import ArmenianBankVectorStore


def _print_results(results: list[dict]) -> None:
    if not results:
        print("  No results above score threshold.")
        return
    for i, r in enumerate(results, 1):
        m = r["metadata"]
        print(f"\n[{i}] {m['bank_name']} · {m['topic']} · score={r['score']:.3f}")
        print(f"    {m['title']}")
        preview = r["text"].replace("\n", " ")[:280]
        print(f"    {preview}{'…' if len(r['text']) > 280 else ''}")


def interactive(store: ArmenianBankVectorStore, bank: str | None, topic: str | None) -> None:
    filters = ", ".join(f"{k}={v}" for k, v in [("bank", bank), ("topic", topic)] if v) or "none"
    print(f"\nKnowledge base query — filters: {filters} — top_k: {RAG_TOP_K}")
    print("Enter a question or 'q' to quit.\n")

    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query or query.lower() in ("q", "quit", "exit"):
            break

        _print_results(store.query(query, top_k=RAG_TOP_K, bank_filter=bank, topic_filter=topic))


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the Armenian bank knowledge base")
    parser.add_argument("--bank", help="Filter by bank ID (ameriabank | ardshinbank | acba)")
    parser.add_argument("--topic", help="Filter by topic (credits | deposits | branch_locations)")
    parser.add_argument("--query", help="Single query (non-interactive)")
    args = parser.parse_args()

    store = ArmenianBankVectorStore()
    n = store.count()
    if n == 0:
        print("Vector store is empty — run: python main.py setup")
        sys.exit(1)

    print(f"Loaded {n} chunks.")

    if args.query:
        _print_results(store.query(args.query, top_k=RAG_TOP_K, bank_filter=args.bank, topic_filter=args.topic))
    else:
        interactive(store, bank=args.bank, topic=args.topic)


if __name__ == "__main__":
    main()
