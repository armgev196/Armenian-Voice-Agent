import base64
import json
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestConfig(unittest.TestCase):
    def test_required_fields_present(self):
        import config
        for attr in [
            "LIVEKIT_URL", "STT_MODEL", "STT_LANGUAGE",
            "LLM_MODEL", "TTS_VOICE_NAME", "EMBEDDING_MODEL",
        ]:
            self.assertIsNotNone(getattr(config, attr), f"{attr} must not be None")

    def test_stt_language_is_armenian(self):
        from config import STT_LANGUAGE
        self.assertEqual(STT_LANGUAGE, "hy")

    def test_tts_voice_is_armenian(self):
        from config import TTS_LANGUAGE_CODE, TTS_VOICE_NAME
        self.assertTrue(TTS_LANGUAGE_CODE.startswith("hy"))
        self.assertIn("hy-AM", TTS_VOICE_NAME)

    def test_allowed_topics_is_set(self):
        from config import ALLOWED_TOPICS
        self.assertIsInstance(ALLOWED_TOPICS, set)
        self.assertIn("credits", ALLOWED_TOPICS)
        self.assertIn("deposits", ALLOWED_TOPICS)
        self.assertIn("branch_locations", ALLOWED_TOPICS)

    def test_rag_bounds(self):
        from config import RAG_MIN_SCORE, RAG_TOP_K
        self.assertGreater(RAG_TOP_K, 0)
        self.assertGreaterEqual(RAG_MIN_SCORE, 0.0)
        self.assertLessEqual(RAG_MIN_SCORE, 1.0)

    def test_banks_registered(self):
        from config import BANKS
        for bank_id in ["ameriabank", "ardshinbank", "acba"]:
            self.assertIn(bank_id, BANKS)
            self.assertIn("name", BANKS[bank_id])
            self.assertIn("base_url", BANKS[bank_id])


class _VectorStoreBase(unittest.TestCase):
    """Mixin that creates an isolated temp ChromaDB for each test class."""

    @classmethod
    def setUpClass(cls):
        import config
        cls.tmpdir = tempfile.mkdtemp()
        cls._orig_chroma = config.CHROMA_DIR
        config.CHROMA_DIR = Path(cls.tmpdir) / "chroma"
        from knowledge_base.vectorstore import ArmenianBankVectorStore
        cls.store = ArmenianBankVectorStore()

    @classmethod
    def tearDownClass(cls):
        import config
        config.CHROMA_DIR = cls._orig_chroma
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    @staticmethod
    def _sample_docs() -> list[dict]:
        return [
            {
                "title": "Ameriabank Consumer Loan",
                "content": (
                    "Ameriabank offers consumer loans from 13% annually. "
                    "Maximum AMD 5,000,000. Term up to 60 months."
                ),
                "url": "https://ameriabank.am/loans/consumer",
                "bank": "ameriabank",
                "bank_name": "Ameriabank",
                "topic": "credits",
            },
            {
                "title": "ACBA Term Deposit",
                "content": (
                    "ACBA term deposits up to 9.5% annual in AMD. "
                    "Minimum AMD 100,000. Terms 3–36 months."
                ),
                "url": "https://acba.am/deposits/term",
                "bank": "acba",
                "bank_name": "ACBA Bank",
                "topic": "deposits",
            },
            {
                "title": "Ardshinbank Yerevan Branches",
                "content": (
                    "Main branch: 13 Vazgen Sargsyan St, Yerevan. "
                    "Mon–Fri 09:00–18:00. Tel: +374 10 560600."
                ),
                "url": "https://ardshinbank.am/branches",
                "bank": "ardshinbank",
                "bank_name": "Ardshinbank",
                "topic": "branch_locations",
            },
        ]


class TestVectorStoreCRUD(_VectorStoreBase):
    def setUp(self):
        self.store.clear()

    def test_add_returns_positive_count(self):
        n = self.store.add_documents(self._sample_docs())
        self.assertGreater(n, 0)

    def test_count_reflects_additions(self):
        self.store.add_documents(self._sample_docs())
        self.assertGreater(self.store.count(), 0)

    def test_clear_empties_store(self):
        self.store.add_documents(self._sample_docs())
        self.store.clear()
        self.assertEqual(self.store.count(), 0)

    def test_stats_shape(self):
        self.store.add_documents(self._sample_docs())
        stats = self.store.stats()
        self.assertIn("total_chunks", stats)
        self.assertIn("collection", stats)
        self.assertGreater(stats["total_chunks"], 0)

    def test_upsert_is_idempotent(self):
        self.store.add_documents(self._sample_docs())
        count_first = self.store.count()
        self.store.add_documents(self._sample_docs())
        self.assertEqual(self.store.count(), count_first)


class TestVectorStoreQuery(_VectorStoreBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.store.clear()
        cls.store.add_documents(cls._sample_docs())

    def test_query_returns_list(self):
        results = self.store.query("loan interest rate", top_k=3)
        self.assertIsInstance(results, list)

    def test_result_has_required_keys(self):
        results = self.store.query("deposit", top_k=1)
        self.assertTrue(len(results) > 0)
        r = results[0]
        self.assertIn("text", r)
        self.assertIn("metadata", r)
        self.assertIn("score", r)

    def test_scores_are_descending(self):
        results = self.store.query("interest rate", top_k=3)
        scores = [r["score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_score_within_bounds(self):
        for r in self.store.query("bank", top_k=5):
            self.assertGreaterEqual(r["score"], 0.0)
            self.assertLessEqual(r["score"], 1.01)

    def test_topic_filter_respected(self):
        results = self.store.query("rate", top_k=5, topic_filter="deposits")
        for r in results:
            self.assertEqual(r["metadata"]["topic"], "deposits")

    def test_bank_filter_respected(self):
        results = self.store.query("branch", top_k=5, bank_filter="ardshinbank")
        for r in results:
            self.assertEqual(r["metadata"]["bank"], "ardshinbank")

    def test_empty_store_returns_empty_list(self):
        self.store.clear()
        self.assertEqual(self.store.query("anything"), [])
        self.store.add_documents(self._sample_docs())


class TestVectorStoreFilters(unittest.TestCase):
    def _build_filter(self, bank=None, topic=None):
        from knowledge_base.vectorstore import ArmenianBankVectorStore
        return ArmenianBankVectorStore._build_filter(bank, topic)

    def test_no_filters_returns_none(self):
        self.assertIsNone(self._build_filter())

    def test_single_filter_is_flat_dict(self):
        f = self._build_filter(bank="ameriabank")
        self.assertEqual(f, {"bank": "ameriabank"})

    def test_two_filters_use_and(self):
        f = self._build_filter(bank="acba", topic="deposits")
        self.assertIn("$and", f)
        self.assertEqual(len(f["$and"]), 2)


class TestPrompts(unittest.TestCase):
    def test_system_prompt_renders_context(self):
        from agent.prompts import SYSTEM_PROMPT_TEMPLATE
        rendered = SYSTEM_PROMPT_TEMPLATE.format(context="TEST_CONTEXT")
        self.assertIn("TEST_CONTEXT", rendered)

    def test_classifier_prompt_renders_query(self):
        from agent.prompts import TOPIC_CLASSIFIER_PROMPT
        rendered = TOPIC_CLASSIFIER_PROMPT.format(query="some question")
        self.assertIn("some question", rendered)
        for category in ["credits", "deposits", "branch_locations", "off_topic"]:
            self.assertIn(category, rendered)

    def test_greeting_non_empty(self):
        from agent.prompts import GREETING_ARMENIAN
        self.assertGreater(len(GREETING_ARMENIAN.strip()), 20)

    def test_fallback_responses_non_empty(self):
        from agent.prompts import NO_CONTEXT_RESPONSE, OUT_OF_SCOPE_RESPONSE
        for resp in [OUT_OF_SCOPE_RESPONSE, NO_CONTEXT_RESPONSE]:
            self.assertGreater(len(resp.strip()), 10)


class TestTopicClassifier(unittest.IsolatedAsyncioTestCase):
    async def _make_llm(self):
        with patch("agent.voice_agent.get_store") as mock_get:
            mock_store = MagicMock()
            mock_store.count.return_value = 10
            mock_get.return_value = mock_store
            from agent.voice_agent import ArmenianBankLLM
            llm = ArmenianBankLLM.__new__(ArmenianBankLLM)
            llm._store = mock_store
            llm._client = AsyncMock()
            return llm

    async def _mock_classify(self, llm, response: str) -> str:
        llm._client.messages.create.return_value = MagicMock(
            content=[MagicMock(text=response)]
        )
        return await llm._classify_topic("test query")

    async def test_credits(self):
        llm = await self._make_llm()
        self.assertEqual(await self._mock_classify(llm, "credits"), "credits")

    async def test_deposits(self):
        llm = await self._make_llm()
        self.assertEqual(await self._mock_classify(llm, "deposits"), "deposits")

    async def test_branch_locations(self):
        llm = await self._make_llm()
        self.assertEqual(await self._mock_classify(llm, "branch_locations"), "branch_locations")

    async def test_off_topic(self):
        llm = await self._make_llm()
        self.assertEqual(await self._mock_classify(llm, "off_topic"), "off_topic")

    async def test_api_error_returns_unknown(self):
        llm = await self._make_llm()
        llm._client.messages.create.side_effect = Exception("network error")
        result = await llm._classify_topic("any query")
        self.assertEqual(result, "unknown")


class TestTokenServer(unittest.TestCase):
    def _decode_payload(self, token: str) -> dict:
        part = token.split(".")[1]
        part += "=" * (4 - len(part) % 4)
        return json.loads(base64.urlsafe_b64decode(part))

    def test_token_is_three_part_jwt(self):
        from token_server import create_livekit_token
        token = create_livekit_token("key", "secret", "room", "user")
        self.assertEqual(len(token.split(".")), 3)

    def test_payload_contains_room_and_identity(self):
        from token_server import create_livekit_token
        token = create_livekit_token("key", "secret", "my-room", "alice")
        payload = self._decode_payload(token)
        self.assertEqual(payload["sub"], "alice")
        self.assertEqual(payload["video"]["room"], "my-room")
        self.assertTrue(payload["video"]["roomJoin"])

    def test_ttl_reflected_in_expiry(self):
        from token_server import create_livekit_token
        token = create_livekit_token("k", "s", "r", "u", ttl_seconds=300)
        payload = self._decode_payload(token)
        self.assertAlmostEqual(payload["exp"] - payload["iat"], 300, delta=2)

    def test_different_secrets_produce_different_signatures(self):
        from token_server import create_livekit_token
        t1 = create_livekit_token("k", "secret-a", "r", "u")
        t2 = create_livekit_token("k", "secret-b", "r", "u")
        self.assertNotEqual(t1.split(".")[2], t2.split(".")[2])


class TestScraperBase(unittest.TestCase):
    def _make_scraper(self):
        from scraper.base_scraper import BaseBankScraper

        class Stub(BaseBankScraper):
            bank_id = "stub"
            bank_name = "Stub Bank"
            base_url = "https://stub.am"
            async def scrape_credits(self): return []
            async def scrape_deposits(self): return []
            async def scrape_branches(self): return []

        return Stub()

    def test_clean_text_collapses_whitespace(self):
        s = self._make_scraper()
        self.assertEqual(s._clean_text("  hello   world  "), "hello world")
        self.assertEqual(s._clean_text("a\n\n\nb"), "a b")
        self.assertEqual(s._clean_text(""), "")

    def test_make_doc_sets_bank_fields(self):
        from scraper.base_scraper import ScrapedDocument
        s = self._make_scraper()
        doc = s._make_doc("T", "C", "https://stub.am/x", "credits")
        self.assertIsInstance(doc, ScrapedDocument)
        self.assertEqual(doc.bank, "stub")
        self.assertEqual(doc.bank_name, "Stub Bank")
        self.assertEqual(doc.topic, "credits")

    def test_scraped_document_to_dict_has_required_keys(self):
        from scraper.base_scraper import ScrapedDocument
        doc = ScrapedDocument(
            title="T", content="C", url="u",
            bank="ameriabank", bank_name="Ameriabank", topic="credits"
        )
        d = doc.to_dict()
        for key in ["title", "content", "url", "bank", "bank_name", "topic"]:
            self.assertIn(key, d)

    def test_extract_tables_parses_html(self):
        from bs4 import BeautifulSoup
        s = self._make_scraper()
        html = "<table><tr><th>Rate</th><th>Term</th></tr><tr><td>12%</td><td>24mo</td></tr></table>"
        soup = BeautifulSoup(html, "lxml")
        tables = s._extract_tables(soup)
        self.assertEqual(len(tables), 1)
        self.assertIn("12%", tables[0])
        self.assertIn("24mo", tables[0])


class TestIngestPipeline(unittest.TestCase):
    def test_ingest_populates_store(self):
        tmpdir = tempfile.mkdtemp()
        try:
            import config
            orig_data, orig_chroma = config.DATA_DIR, config.CHROMA_DIR
            config.DATA_DIR = Path(tmpdir)
            config.CHROMA_DIR = Path(tmpdir) / "chroma"

            sample = [{
                "title": "Sample Loan",
                "content": "Interest rate 12% per year for up to 36 months.",
                "url": "https://ameriabank.am/loans",
                "bank": "ameriabank",
                "bank_name": "Ameriabank",
                "topic": "credits",
            }]
            with open(Path(tmpdir) / "scraped_data.json", "w") as f:
                json.dump(sample, f)

            from knowledge_base.ingest import main as ingest_main
            ingest_main(clear=True)

            from knowledge_base.vectorstore import ArmenianBankVectorStore
            self.assertGreater(ArmenianBankVectorStore().count(), 0)

        finally:
            config.DATA_DIR = orig_data
            config.CHROMA_DIR = orig_chroma
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_ingest_skips_short_content(self):
        tmpdir = tempfile.mkdtemp()
        try:
            import config
            orig_data, orig_chroma = config.DATA_DIR, config.CHROMA_DIR
            config.DATA_DIR = Path(tmpdir)
            config.CHROMA_DIR = Path(tmpdir) / "chroma2"

            sample = [
                {"title": "A", "content": "too short", "url": "u",
                 "bank": "ameriabank", "bank_name": "Ameriabank", "topic": "credits"},
            ]
            with open(Path(tmpdir) / "scraped_data.json", "w") as f:
                json.dump(sample, f)

            from knowledge_base.ingest import main as ingest_main
            ingest_main(clear=True)

            from knowledge_base.vectorstore import ArmenianBankVectorStore
            self.assertEqual(ArmenianBankVectorStore().count(), 0)

        finally:
            config.DATA_DIR = orig_data
            config.CHROMA_DIR = orig_chroma
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
