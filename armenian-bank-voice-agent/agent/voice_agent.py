"""
agent/voice_agent.py — LiveKit Voice Pipeline Agent for Armenian bank support.

Tech Stack Rationale:
──────────────────────────────────────────────────────────────────────────────
STT: OpenAI Whisper (whisper-1)
  WHY: Best-in-class support for Armenian ("hy"). Tested against Google STT,
  Deepgram, and Azure — Whisper has the highest WER accuracy on Armenian text,
  especially with bank/finance vocabulary. The language=hy parameter explicitly
  forces Armenian decoding, avoiding mis-identification as other languages.

LLM: Anthropic Claude (claude-sonnet-4-20250514)
  WHY: Claude's system prompt adherence is critical here — we need the model to
  STRICTLY refuse to answer from outside knowledge. Claude's instruction-following
  is superior to GPT-4o for constrained RAG scenarios. It also produces natural
  conversational Armenian without code-switching to English.

TTS: Google Cloud Text-to-Speech (hy-AM-Standard-A)
  WHY: The only production-quality Armenian TTS voice available. ElevenLabs and
  Azure have no Armenian voice. OpenAI TTS produces English accent when given
  Armenian text. Google hy-AM-Standard-A is native-sounding and handles
  Armenian punctuation (։) correctly.

VAD: Silero VAD
  WHY: Lightweight, runs locally, excellent sensitivity. Better than WebRTC VAD
  for noisy environments (common in bank branch/call center settings).

Vector DB: ChromaDB + paraphrase-multilingual-mpnet-base-v2
  WHY: Multilingual sentence transformer explicitly supports Armenian.
  ChromaDB is zero-infrastructure, persists locally — no cloud dependency.
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import logging
import os
import asyncio
from typing import AsyncIterator

from livekit import agents
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm as agents_llm,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import openai as lk_openai, google as lk_google, silero

import anthropic

from config import (
    LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET,
    STT_MODEL, STT_LANGUAGE,
    LLM_MODEL, LLM_MAX_TOKENS,
    TTS_LANGUAGE_CODE, TTS_VOICE_NAME, TTS_SPEAKING_RATE,
    RAG_TOP_K, ANTHROPIC_API_KEY,
)
from knowledge_base.vectorstore import ArmenianBankVectorStore
from agent.prompts import (
    SYSTEM_PROMPT_TEMPLATE,
    GREETING_ARMENIAN,
    TOPIC_CLASSIFIER_PROMPT,
    OUT_OF_SCOPE_RESPONSE,
    NO_CONTEXT_RESPONSE,
)

logger = logging.getLogger(__name__)

# ── Shared Vector Store (loaded once, reused across sessions) ──────────────────
_vector_store: ArmenianBankVectorStore | None = None


def get_vector_store() -> ArmenianBankVectorStore:
    global _vector_store
    if _vector_store is None:
        logger.info("Initializing vector store...")
        _vector_store = ArmenianBankVectorStore()
        count = _vector_store.count()
        if count == 0:
            logger.warning(
                "Vector store is empty! Run: python -m scraper.run_all && "
                "python -m knowledge_base.ingest"
            )
        else:
            logger.info(f"Vector store ready: {count} chunks")
    return _vector_store


# ── RAG-Augmented Claude LLM ───────────────────────────────────────────────────

class ArmenianBankLLM(agents_llm.LLM):
    """
    Custom LLM wrapper that:
    1. Classifies the user query topic
    2. Retrieves relevant bank knowledge chunks (RAG)
    3. Calls Claude with strict context-only instructions
    4. Returns streaming response
    """

    def __init__(self):
        super().__init__()
        self._client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        self._store  = get_vector_store()

    async def _classify_topic(self, query: str) -> str:
        """Classify query as credits/deposits/branch_locations/off_topic."""
        try:
            resp = await self._client.messages.create(
                model=LLM_MODEL,
                max_tokens=20,
                messages=[{
                    "role": "user",
                    "content": TOPIC_CLASSIFIER_PROMPT.format(query=query)
                }],
            )
            return resp.content[0].text.strip().lower()
        except Exception as e:
            logger.warning(f"Topic classification failed: {e}")
            return "unknown"

    async def _retrieve_context(self, query: str, topic: str) -> str:
        """Retrieve relevant RAG chunks and format as context string."""
        topic_filter = topic if topic in ["credits", "deposits", "branch_locations"] else None

        results = self._store.query(
            query_text=query,
            top_k=RAG_TOP_K,
            topic_filter=topic_filter,
        )

        if not results:
            return ""

        parts = []
        for r in results:
            meta = r["metadata"]
            parts.append(
                f"[{meta['bank_name']} | {meta['topic']} | score: {r['score']:.2f}]\n"
                f"{r['text']}"
            )
        return "\n\n---\n\n".join(parts)

    async def chat(
        self,
        chat_ctx: agents_llm.ChatContext,
        **kwargs,
    ) -> agents_llm.LLMStream:
        """Main entry point called by VoicePipelineAgent after each STT result."""

        # Extract the latest user message
        user_message = ""
        for msg in reversed(chat_ctx.messages):
            if msg.role == "user" and isinstance(msg.content, str):
                user_message = msg.content
                break

        if not user_message:
            return self._empty_stream()

        # 1. Classify topic
        topic = await self._classify_topic(user_message)
        logger.info(f"Query: '{user_message[:80]}' → topic: {topic}")

        # 2. Handle off-topic queries immediately
        if topic == "off_topic":
            return self._static_stream(OUT_OF_SCOPE_RESPONSE)

        # 3. Retrieve relevant context
        context = await self._retrieve_context(user_message, topic)

        if not context:
            return self._static_stream(NO_CONTEXT_RESPONSE)

        # 4. Build system prompt with injected context
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)

        # 5. Build conversation history (last 6 turns to stay within token budget)
        messages = []
        for msg in chat_ctx.messages[-6:]:
            if isinstance(msg.content, str) and msg.content.strip():
                role = "user" if msg.role == "user" else "assistant"
                messages.append({"role": role, "content": msg.content})

        # Ensure we end on a user message
        if not messages or messages[-1]["role"] != "user":
            messages.append({"role": "user", "content": user_message})

        # 6. Stream Claude response
        return ClaudeStreamWrapper(
            client=self._client,
            system=system_prompt,
            messages=messages,
        )

    @staticmethod
    def _static_stream(text: str) -> "StaticLLMStream":
        return StaticLLMStream(text)

    @staticmethod
    def _empty_stream() -> "StaticLLMStream":
        return StaticLLMStream("")


class StaticLLMStream(agents_llm.LLMStream):
    """Returns a pre-defined static text response (for off-topic / no-context)."""

    def __init__(self, text: str):
        super().__init__(None, chat_ctx=None, fnc_ctx=None)
        self._text = text

    async def aclose(self):
        pass

    async def __anext__(self) -> agents_llm.ChatChunk:
        if self._text:
            chunk = agents_llm.ChatChunk(
                request_id="static",
                choices=[agents_llm.Choice(
                    delta=agents_llm.ChoiceDelta(role="assistant", content=self._text),
                    index=0,
                )],
            )
            self._text = ""
            return chunk
        raise StopAsyncIteration


class ClaudeStreamWrapper(agents_llm.LLMStream):
    """Wraps Anthropic streaming API into LiveKit LLMStream interface."""

    def __init__(self, client: anthropic.AsyncAnthropic, system: str, messages: list):
        super().__init__(None, chat_ctx=None, fnc_ctx=None)
        self._client   = client
        self._system   = system
        self._messages = messages
        self._stream   = None
        self._buffer   = asyncio.Queue()
        self._done     = False

    async def _start_stream(self):
        """Begin the Anthropic stream in the background."""
        try:
            async with self._client.messages.stream(
                model=LLM_MODEL,
                max_tokens=LLM_MAX_TOKENS,
                system=self._system,
                messages=self._messages,
            ) as stream:
                async for text in stream.text_stream:
                    await self._buffer.put(text)
        except Exception as e:
            logger.error(f"Claude streaming error: {e}")
        finally:
            self._done = True
            await self._buffer.put(None)  # sentinel

    async def aclose(self):
        self._done = True

    async def __anext__(self) -> agents_llm.ChatChunk:
        if self._stream is None:
            # Kick off the stream on first call
            self._stream = asyncio.create_task(self._start_stream())

        text = await self._buffer.get()
        if text is None:
            raise StopAsyncIteration

        return agents_llm.ChatChunk(
            request_id="claude",
            choices=[agents_llm.Choice(
                delta=agents_llm.ChoiceDelta(role="assistant", content=text),
                index=0,
            )],
        )


# ── LiveKit Agent Job ──────────────────────────────────────────────────────────

async def entrypoint(ctx: JobContext):
    """Called by LiveKit workers for each new room/call."""
    logger.info(f"Starting Armenian Bank Voice Agent in room: {ctx.room.name}")

    # Pre-load vector store
    get_vector_store()

    # STT: OpenAI Whisper with forced Armenian language
    stt = lk_openai.STT(
        model=STT_MODEL,
        language=STT_LANGUAGE,      # "hy" = Armenian
    )

    # LLM: Our custom RAG-augmented Claude wrapper
    llm_instance = ArmenianBankLLM()

    # TTS: Google Cloud with Armenian voice
    tts = lk_google.TTS(
        language=TTS_LANGUAGE_CODE,     # "hy-AM"
        voice_name=TTS_VOICE_NAME,      # "hy-AM-Standard-A"
        speaking_rate=TTS_SPEAKING_RATE,
    )

    # VAD: Silero (local, no API key needed)
    vad = silero.VAD.load(
        min_silence_duration=0.5,   # seconds of silence to end speech
        min_speech_duration=0.1,    # minimum speech burst
    )

    # Assemble the pipeline
    agent = VoicePipelineAgent(
        vad=vad,
        stt=stt,
        llm=llm_instance,
        tts=tts,
        min_endpointing_delay=0.6,
        max_endpointing_delay=5.0,
    )

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    agent.start(ctx.room)

    # Greet the user in Armenian
    await agent.say(GREETING_ARMENIAN, allow_interruptions=True)


# ── Worker Entry Point ─────────────────────────────────────────────────────────

def run_worker():
    """Start the LiveKit worker process."""
    os.environ.setdefault("LIVEKIT_URL", LIVEKIT_URL)
    os.environ.setdefault("LIVEKIT_API_KEY", LIVEKIT_API_KEY)
    os.environ.setdefault("LIVEKIT_API_SECRET", LIVEKIT_API_SECRET)

    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
