"""
tests/test_all.py — Test suite for the Armenian Bank Voice Agent.

Run from project root:
    python -m pytest tests/ -v
    python -m pytest tests/ -v -k "rag"         # only RAG tests
    python -m pytest tests/ -v -k "scraper"     # only scraper tests
    python -m pytest tests/ -v --no-header -q   # quiet mode

Tests are grouped into:
  - TestConfig         — environment and config loading
  - TestVectorStore    — ChromaDB insert & query
  - TestRAGPipeline    — full RAG retrieval flow
  - TestPrompts        — prompt template rendering
  - TestTopicClassifier— topic classification logic
  - TestTokenServer    — JWT token generation
  - TestScraper        — scraper base class + HTML parsing
"""

import json
import os
import sys
import time
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ══════════════════════════════════════════════════════════════════════════════
# 1. Config Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestConfig(unittest.TestCase):
    """Verify config loads without crashing and has required keys."""

    def test_config_imports(self):
        import config
        self.assertIsNotNone(config.LIVEKIT_URL)
        self.assertIsNotNone(config.STT_MODEL)
        self.assertIsNotNone(config.LLM_MODEL)
        self.assertIsNotNone(config.TTS_VOICE_NAME)
        self.assertIsNotNone(config.EMBEDDING_MODEL)

    def test_banks_dict(self):
        from config import BANKS, ALLOWED_TOPICS
        self.assertIn("ameriabank",  BANKS)
        self.assertIn("ardshinbank", BANKS)
        self.assertIn("acba",        BANKS)
        self.assertIn("credits",     ALLOWED_TOPICS)
        self.assertIn("deposits",    ALLOWED_TOPICS)
        self.assertIn("branch_locations", ALLOWED_TOPICS)

    def test_rag_settings(self):
        from config import RAG_TOP_K, RAG_MIN_SCORE
        self.assertGreater(RAG_TOP_K, 0)
        self.assertGreaterEqual(RAG_MIN_SCORE, 0.0)
        self.assertLessEqual(RAG_MIN_SCORE,    1.0)

    def test_stt_language_is_armenian(self):
        from config import STT_LANGUAGE
        self.assertEqual(STT_LANGUAGE, "hy",
            "STT language must be 'hy' (Armenian) for correct Whisper transcription")

    def test_tts_is_armenian(self):
        from config import TTS_LANGUAGE_CODE, TTS_VOICE_NAME
        self.assertTrue(TTS_LANGUAGE_CODE.startswith("hy"),
            f"TTS language must start with 'hy', got: {TTS_LANGUAGE_CODE}")
        self.assertIn("hy-AM", TTS_VOICE_NAME,
            f"TTS voice must be Armenian, got: {TTS_VOICE_NAME}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Vector Store Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestVectorStore(unittest.TestCase):
    """Test ChromaDB CRUD and retrieval using a temporary directory."""

    @classmethod
    def setUpClass(cls):
        """Create an isolated vector store for testing."""
        cls.tmpdir = tempfile.mkdtemp()
        # Patch CHROMA_DIR before importing vectorstore
        import config
        cls._orig_chroma_dir = config.CHROMA_DIR
        config.CHROMA_DIR = Path(cls.tmpdir) / "test_chroma"

        from knowledge_base.vectorstore import ArmenianBankVectorStore
        cls.store = ArmenianBankVectorStore()

    @classmethod
    def tearDownClass(cls):
        import config
        config.CHROMA_DIR = cls._orig_chroma_dir
        import shutil
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def _sample_documents(self) -> list[dict]:
        return [
            {
                "title":     "Ameriabank Consumer Loan",
                "content":   "Ameriabank offers consumer loans with interest rates starting from 13% annually. "
                             "Maximum loan amount is AMD 5,000,000. Loan term up to 60 months.",
                "url":       "https://ameriabank.am/loans/consumer",
                "bank":      "ameriabank",
                "bank_name": "Ameriabank",
                "topic":     "credits",
            },
            {
                "title":     "ACBA Term Deposit",
                "content":   "ACBA Bank term deposits offer up to 9.5% annual interest rate in AMD. "
                             "Minimum deposit amount AMD 100,000. Terms from 3 to 36 months.",
                "url":       "https://acba.am/deposits/term",
                "bank":      "acba",
                "bank_name": "ACBA Bank",
                "topic":     "deposits",
            },
            {
                "title":     "Ardshinbank Yerevan Branches",
                "content":   "Ardshinbank main branch: 13 Vazgen Sargsyan Street, Yerevan. "
                             "Open Mon-Fri 09:00-18:00. Phone: +374 10 560600.",
                "url":       "https://ardshinbank.am/branches",
                "bank":      "ardshinbank",
                "bank_name": "Ardshinbank",
                "topic":     "branch_locations",
            },
        ]

    def test_add_and_count(self):
        self.store.clear()
        n = self.store.add_documents(self._sample_documents())
        self.assertGreater(n, 0, "Should have added at least one chunk")
        self.assertGreater(self.store.count(), 0)

    def test_query_returns_results(self):
        self.store.clear()
        self.store.add_documents(self._sample_documents())
        results = self.store.query("What are the interest rates for loans?", top_k=3)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0, "Should return at least one result")

    def test_query_result_structure(self):
        results = self.store.query("deposit interest rate", top_k=2)
        if results:
            r = results[0]
            self.assertIn("text",     r)
            self.assertIn("metadata", r)
            self.assertIn("score",    r)
            self.assertGreater(r["score"], 0.0)
            self.assertLessEqual(r["score"], 1.01)

    def test_topic_filter(self):
        results = self.store.query("interest rate", top_k=5, topic_filter="deposits")
        for r in results:
            self.assertEqual(r["metadata"]["topic"], "deposits",
                             "Topic filter should only return deposit chunks")

    def test_bank_filter(self):
        results = self.store.query("branch address", top_k=5, bank_filter="ardshinbank")
        for r in results:
            self.assertEqual(r["metadata"]["bank"], "ardshinbank",
                             "Bank filter should only return ardshinbank chunks")

    def test_clear(self):
        self.store.add_documents(self._sample_documents())
        self.store.clear()
        self.assertEqual(self.store.count(), 0)

    def test_empty_query_returns_empty(self):
        # Empty store should not crash
        self.store.clear()
        results = self.store.query("loans", top_k=5)
        self.assertIsInstance(results, list)

    def test_stats(self):
        self.store.clear()
        self.store.add_documents(self._sample_documents())
        stats = self.store.stats()
        self.assertIn("total_chunks", stats)
        self.assertGreater(stats["total_chunks"], 0)


# ══════════════════════════════════════════════════════════════════════════════
# 3. RAG Pipeline Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestRAGPipeline(unittest.TestCase):
    """Test that the RAG pipeline finds semantically relevant results."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        import config
        cls._orig = config.CHROMA_DIR
        config.CHROMA_DIR = Path(cls.tmpdir) / "rag_test"

        from knowledge_base.vectorstore import ArmenianBankVectorStore
        cls.store = ArmenianBankVectorStore()
        cls.store.add_documents([
            {
                "title": "Mortgage Loan", "bank": "ameriabank", "bank_name": "Ameriabank",
                "topic": "credits", "url": "https://ameriabank.am/mortgage",
                "content": "Ameriabank mortgage loans for purchasing real estate. "
                           "Interest rate 10.5% annual. Term up to 20 years. "
                           "Maximum loan 50,000,000 AMD.",
            },
            {
                "title": "Savings Account", "bank": "acba", "bank_name": "ACBA Bank",
                "topic": "deposits", "url": "https://acba.am/savings",
                "content": "ACBA savings account with 7% annual interest. "
                           "No minimum balance required. Access anytime.",
            },
            {
                "title": "Branch Kentron", "bank": "ameriabank", "bank_name": "Ameriabank",
                "topic": "branch_locations", "url": "https://ameriabank.am/branches",
                "content": "Ameriabank Kentron branch at 9 Grigor Lusavorich Street, Yerevan. "
                           "Working hours: Mon-Sat 09:00-19:00.",
            },
        ])

    @classmethod
    def tearDownClass(cls):
        import config
        config.CHROMA_DIR = cls._orig
        import shutil
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_mortgage_query_finds_credits(self):
        results = self.store.query("I want to buy an apartment, what loans are available?", top_k=3)
        topics = [r["metadata"]["topic"] for r in results]
        self.assertIn("credits", topics, "Mortgage query should surface credits topic")

    def test_branch_query_finds_locations(self):
        results = self.store.query("Where is Ameriabank located in Yerevan?", top_k=3)
        self.assertGreater(len(results), 0)
        topics = [r["metadata"]["topic"] for r in results]
        self.assertIn("branch_locations", topics)

    def test_armenian_query_works(self):
        """Multilingual embedding model must handle Armenian text."""
        results = self.store.query("Ինչ տոկոսադրույք ունի հիփոթեքային վarkel?", top_k=3)
        self.assertIsInstance(results, list)  # Should not crash, even if no results

    def test_scores_are_descending(self):
        results = self.store.query("interest rate deposit savings", top_k=3)
        if len(results) > 1:
            scores = [r["score"] for r in results]
            self.assertEqual(scores, sorted(scores, reverse=True),
                             "Results should be sorted by score descending")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Prompt Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestPrompts(unittest.TestCase):

    def test_system_prompt_renders(self):
        from agent.prompts import SYSTEM_PROMPT_TEMPLATE
        rendered = SYSTEM_PROMPT_TEMPLATE.format(context="Test context here.")
        self.assertIn("Test context here.", rendered)
        self.assertIn("հայerên", rendered.lower().replace("հայerên", "հայerên")
                      or rendered)  # Armenian instruction must be present

    def test_system_prompt_has_armenian(self):
        from agent.prompts import SYSTEM_PROMPT_TEMPLATE
        # Must contain Armenian Unicode characters
        self.assertTrue(
            any('\u0531' <= c <= '\u058F' for c in SYSTEM_PROMPT_TEMPLATE),
            "System prompt must contain Armenian characters"
        )

    def test_greeting_is_armenian(self):
        from agent.prompts import GREETING_ARMENIAN
        self.assertTrue(len(GREETING_ARMENIAN) > 20)
        has_armenian = any('\u0531' <= c <= '\u058F' for c in GREETING_ARMENIAN)
        self.assertTrue(has_armenian, "Greeting must contain Armenian characters")

    def test_out_of_scope_response_is_armenian(self):
        from agent.prompts import OUT_OF_SCOPE_RESPONSE
        has_armenian = any('\u0531' <= c <= '\u058F' for c in OUT_OF_SCOPE_RESPONSE)
        self.assertTrue(has_armenian)

    def test_topic_classifier_prompt_has_placeholder(self):
        from agent.prompts import TOPIC_CLASSIFIER_PROMPT
        self.assertIn("{query}", TOPIC_CLASSIFIER_PROMPT)
        rendered = TOPIC_CLASSIFIER_PROMPT.format(query="test query")
        self.assertIn("test query", rendered)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Topic Classifier Tests (mocked — no real API calls)
# ══════════════════════════════════════════════════════════════════════════════

class TestTopicClassifier(unittest.IsolatedAsyncioTestCase):

    async def _make_llm(self):
        """Create ArmenianBankLLM with mocked vector store and Anthropic client."""
        with patch("agent.voice_agent.get_vector_store") as mock_vs:
            mock_store = MagicMock()
            mock_store.count.return_value = 10
            mock_vs.return_value = mock_store

            from agent.voice_agent import ArmenianBankLLM
            llm = ArmenianBankLLM.__new__(ArmenianBankLLM)
            llm._store = mock_store

            mock_client = AsyncMock()
            llm._client = mock_client
            return llm, mock_client

    async def test_classify_credits(self):
        llm, mock_client = await self._make_llm()
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="credits")]
        )
        result = await llm._classify_topic("What loans does Ameriabank offer?")
        self.assertEqual(result, "credits")

    async def test_classify_deposits(self):
        llm, mock_client = await self._make_llm()
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="deposits")]
        )
        result = await llm._classify_topic("What is the deposit interest rate?")
        self.assertEqual(result, "deposits")

    async def test_classify_off_topic(self):
        llm, mock_client = await self._make_llm()
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="off_topic")]
        )
        result = await llm._classify_topic("What is the capital of France?")
        self.assertEqual(result, "off_topic")

    async def test_classify_fallback_on_error(self):
        llm, mock_client = await self._make_llm()
        mock_client.messages.create.side_effect = Exception("API error")
        result = await llm._classify_topic("any query")
        self.assertEqual(result, "unknown")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Token Server Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestTokenServer(unittest.TestCase):

    def test_create_token_structure(self):
        from token_server import create_livekit_token
        token = create_livekit_token(
            api_key="testkey", api_secret="testsecret",
            room="test-room", identity="test-user",
        )
        parts = token.split(".")
        self.assertEqual(len(parts), 3, "JWT must have 3 parts: header.payload.signature")

    def test_create_token_is_string(self):
        from token_server import create_livekit_token
        token = create_livekit_token("k", "s", "room", "user")
        self.assertIsInstance(token, str)
        self.assertGreater(len(token), 50)

    def test_token_payload_contains_room(self):
        import base64
        import json
        from token_server import create_livekit_token

        token = create_livekit_token("k", "secret", "my-room", "my-user")
        payload_b64 = token.split(".")[1]
        # Add padding
        payload_b64 += "=" * (4 - len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))

        self.assertEqual(payload["sub"], "my-user")
        self.assertEqual(payload["video"]["room"], "my-room")
        self.assertTrue(payload["video"]["roomJoin"])

    def test_token_expiry(self):
        import base64
        import json
        from token_server import create_livekit_token

        token = create_livekit_token("k", "s", "r", "u", ttl_seconds=600)
        payload_b64 = token.split(".")[1] + "=="
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))

        self.assertIn("exp", payload)
        self.assertIn("iat", payload)
        self.assertAlmostEqual(payload["exp"] - payload["iat"], 600, delta=2)


# ══════════════════════════════════════════════════════════════════════════════
# 7. Scraper Base Class Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestScraperBase(unittest.TestCase):

    def test_clean_text(self):
        from scraper.base_scraper import BaseBankScraper

        class _Concrete(BaseBankScraper):
            bank_id = "test"; bank_name = "Test"; base_url = ""
            async def scrape_credits(self): return []
            async def scrape_deposits(self): return []
            async def scrape_branches(self): return []

        s = _Concrete()
        self.assertEqual(s._clean_text("  hello   world  "), "hello world")
        self.assertEqual(s._clean_text("line1\n\n\nline2"), "line1 line2")
        self.assertEqual(s._clean_text(""), "")

    def test_make_doc_structure(self):
        from scraper.base_scraper import BaseBankScraper, ScrapedDocument

        class _Concrete(BaseBankScraper):
            bank_id = "ameriabank"; bank_name = "Ameriabank"; base_url = ""
            async def scrape_credits(self): return []
            async def scrape_deposits(self): return []
            async def scrape_branches(self): return []

        s = _Concrete()
        doc = s._make_doc("Title", "Content", "https://example.com", "credits")
        self.assertIsInstance(doc, ScrapedDocument)
        self.assertEqual(doc.bank, "ameriabank")
        self.assertEqual(doc.topic, "credits")

    def test_scraped_document_to_dict(self):
        from scraper.base_scraper import ScrapedDocument
        doc = ScrapedDocument(
            title="Test", content="Content", url="http://x.com",
            bank="ameriabank", bank_name="Ameriabank", topic="credits",
        )
        d = doc.to_dict()
        self.assertIn("title", d)
        self.assertIn("content", d)
        self.assertIn("bank", d)
        self.assertIn("topic", d)


# ══════════════════════════════════════════════════════════════════════════════
# 8. Integration: Ingest Pipeline
# ══════════════════════════════════════════════════════════════════════════════

class TestIngestPipeline(unittest.TestCase):
    """Test the ingest pipeline with sample data (no scraping needed)."""

    def test_ingest_from_sample_data(self):
        import tempfile, json, shutil
        from pathlib import Path

        tmpdir = tempfile.mkdtemp()
        try:
            import config
            orig_data = config.DATA_DIR
            orig_chroma = config.CHROMA_DIR
            config.DATA_DIR = Path(tmpdir)
            config.CHROMA_DIR = Path(tmpdir) / "chroma"

            # Write sample scraped data
            sample = [
                {
                    "title": "Sample Loan", "content": "Interest rate 12% per year.",
                    "url": "http://x.com", "bank": "ameriabank",
                    "bank_name": "Ameriabank", "topic": "credits",
                }
            ]
            with open(Path(tmpdir) / "scraped_data.json", "w") as f:
                json.dump(sample, f)

            from knowledge_base.ingest import main as ingest_main
            ingest_main(clear=True)   # Should not raise

            from knowledge_base.vectorstore import ArmenianBankVectorStore
            store = ArmenianBankVectorStore()
            self.assertGreater(store.count(), 0)

        finally:
            config.DATA_DIR = orig_data
            config.CHROMA_DIR = orig_chroma
            shutil.rmtree(tmpdir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
