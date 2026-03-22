from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

import anthropic
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm as agents_llm
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import google as lk_google
from livekit.plugins import openai as lk_openai
from livekit.plugins import silero

from agent.prompts import (
    GREETING_ARMENIAN,
    NO_CONTEXT_RESPONSE,
    OUT_OF_SCOPE_RESPONSE,
    SYSTEM_PROMPT_TEMPLATE,
    TOPIC_CLASSIFIER_PROMPT,
)
from config import (
    ALLOWED_TOPICS,
    ANTHROPIC_API_KEY,
    LIVEKIT_API_KEY,
    LIVEKIT_API_SECRET,
    LIVEKIT_URL,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    RAG_TOP_K,
    STT_LANGUAGE,
    STT_MODEL,
    TTS_LANGUAGE_CODE,
    TTS_SPEAKING_RATE,
    TTS_VOICE_NAME,
)
from knowledge_base.vectorstore import ArmenianBankVectorStore

logger = logging.getLogger(__name__)

_store: Optional[ArmenianBankVectorStore] = None


def get_store() -> ArmenianBankVectorStore:
    global _store
    if _store is None:
        _store = ArmenianBankVectorStore()
        n = _store.count()
        if n == 0:
            logger.warning("Vector store is empty — run: python main.py setup")
        else:
            logger.info("Vector store ready (%d chunks)", n)
    return _store


class ArmenianBankLLM(agents_llm.LLM):
    def __init__(self) -> None:
        super().__init__()
        self._client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        self._store = get_store()

    async def _classify_topic(self, query: str) -> str:
        try:
            resp = await self._client.messages.create(
                model=LLM_MODEL,
                max_tokens=20,
                messages=[{"role": "user", "content": TOPIC_CLASSIFIER_PROMPT.format(query=query)}],
            )
            return resp.content[0].text.strip().lower()
        except Exception as e:
            logger.warning("Topic classification failed: %s", e)
            return "unknown"

    async def _build_context(self, query: str, topic: str) -> str:
        topic_filter = topic if topic in ALLOWED_TOPICS else None
        results = self._store.query(query_text=query, top_k=RAG_TOP_K, topic_filter=topic_filter)
        if not results:
            return ""
        parts = [
            f"[{r['metadata']['bank_name']} | {r['metadata']['topic']} | {r['score']:.2f}]\n{r['text']}"
            for r in results
        ]
        return "\n\n---\n\n".join(parts)

    async def chat(self, chat_ctx: agents_llm.ChatContext, **kwargs) -> agents_llm.LLMStream:
        user_message = next(
            (m.content for m in reversed(chat_ctx.messages)
             if m.role == "user" and isinstance(m.content, str)),
            "",
        )

        if not user_message:
            return _StaticStream("")

        topic = await self._classify_topic(user_message)
        logger.info("Query classified as: %s", topic)

        if topic == "off_topic":
            return _StaticStream(OUT_OF_SCOPE_RESPONSE)

        context = await self._build_context(user_message, topic)
        if not context:
            return _StaticStream(NO_CONTEXT_RESPONSE)

        messages = [
            {"role": "user" if m.role == "user" else "assistant", "content": m.content}
            for m in chat_ctx.messages[-6:]
            if isinstance(m.content, str) and m.content.strip()
        ]

        if not messages or messages[-1]["role"] != "user":
            messages.append({"role": "user", "content": user_message})

        return _ClaudeStream(
            client=self._client,
            system=SYSTEM_PROMPT_TEMPLATE.format(context=context),
            messages=messages,
        )


class _StaticStream(agents_llm.LLMStream):
    def __init__(self, text: str) -> None:
        super().__init__(None, chat_ctx=None, fnc_ctx=None)
        self._text = text

    async def aclose(self) -> None:
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


class _ClaudeStream(agents_llm.LLMStream):
    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        system: str,
        messages: list,
    ) -> None:
        super().__init__(None, chat_ctx=None, fnc_ctx=None)
        self._client = client
        self._system = system
        self._messages = messages
        self._task: Optional[asyncio.Task] = None
        self._queue: asyncio.Queue = asyncio.Queue()

    async def _stream_to_queue(self) -> None:
        try:
            async with self._client.messages.stream(
                model=LLM_MODEL,
                max_tokens=LLM_MAX_TOKENS,
                system=self._system,
                messages=self._messages,
            ) as stream:
                async for text in stream.text_stream:
                    await self._queue.put(text)
        except Exception as e:
            logger.error("Streaming error: %s", e)
        finally:
            await self._queue.put(None)

    async def aclose(self) -> None:
        if self._task:
            self._task.cancel()

    async def __anext__(self) -> agents_llm.ChatChunk:
        if self._task is None:
            self._task = asyncio.create_task(self._stream_to_queue())

        text = await self._queue.get()
        if text is None:
            raise StopAsyncIteration

        return agents_llm.ChatChunk(
            request_id="claude",
            choices=[agents_llm.Choice(
                delta=agents_llm.ChoiceDelta(role="assistant", content=text),
                index=0,
            )],
        )


async def entrypoint(ctx: JobContext) -> None:
    logger.info("Agent starting in room: %s", ctx.room.name)

    get_store()

    vad = silero.VAD.load(min_silence_duration=0.5, min_speech_duration=0.1)
    stt = lk_openai.STT(model=STT_MODEL, language=STT_LANGUAGE)
    tts = lk_google.TTS(
        language=TTS_LANGUAGE_CODE,
        voice_name=TTS_VOICE_NAME,
        speaking_rate=TTS_SPEAKING_RATE,
    )

    agent = VoicePipelineAgent(
        vad=vad,
        stt=stt,
        llm=ArmenianBankLLM(),
        tts=tts,
        min_endpointing_delay=0.6,
        max_endpointing_delay=5.0,
    )

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    agent.start(ctx.room)
    await agent.say(GREETING_ARMENIAN, allow_interruptions=True)


def run_worker() -> None:
    os.environ.setdefault("LIVEKIT_URL", LIVEKIT_URL)
    os.environ.setdefault("LIVEKIT_API_KEY", LIVEKIT_API_KEY)
    os.environ.setdefault("LIVEKIT_API_SECRET", LIVEKIT_API_SECRET)
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
