"""
LLM Interface — Provider-Agnostic Abstraction with Fallback Chain

Fallback order (configured in RAGConfig.LLM_FALLBACK_CHAIN):
    1. gemini-3-flash      (primary)
    2. gemini-2.5-flash    (fallback on 429 / failure)
    3. gemma3:27b (Ollama) (fully local, final fallback)

On a 429 rate-limit the engine:
    • Retries the *same* model once after the delay Gemini returns (RPM limit).
    • If the daily quota is exhausted, immediately cascades to the next model.
    • Exhausted models are remembered for the session so we never reroute back.

Providers: "gemini" | "openai" | "anthropic" | "ollama"
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Any, Dict, List, Optional, Set

from .config import RAGConfig


class _RateLimiter:
    """
    Token bucket that enforces a maximum call rate by spacing requests evenly.

    At 5 RPM the minimum inter-call gap is 60 / 5 = 12 seconds.  Each call to
    acquire() blocks the calling thread until that gap has elapsed since the
    previous call.  This is proactive — we never send a request that would
    arrive at the API before the rate window is available.
    """

    def __init__(self, calls_per_minute: int) -> None:
        self._gap       = 60.0 / max(calls_per_minute, 1)   # seconds between calls
        self._lock      = threading.Lock()
        self._last_call = 0.0   # monotonic timestamp of last acquired slot

    def acquire(self) -> None:
        with self._lock:
            now     = time.monotonic()
            elapsed = now - self._last_call
            wait    = self._gap - elapsed
            if wait > 0:
                logging.getLogger(__name__).info(
                    "Rate limiter: waiting %.1fs before next LLM call (%.0f RPM cap)",
                    wait, 60.0 / self._gap,
                )
                time.sleep(wait)
            self._last_call = time.monotonic()

logger = logging.getLogger(__name__)


# Module-level rate limiter — shared across all LLMInterface instances.
# Instantiated lazily when first LLMInterface is created so RAGConfig is
# already fully loaded.
_rate_limiter: Optional["_RateLimiter"] = None


class LLMInterface:
    """
    Cascading LLM adapter.  Tries each entry in LLM_FALLBACK_CHAIN in order.
    Exhausted models are skipped for the lifetime of the process.
    """

    def __init__(self) -> None:
        global _rate_limiter
        if _rate_limiter is None:
            _rate_limiter = _RateLimiter(RAGConfig.LLM_RPM_CAP)
        # Lazily-built LLM instances keyed by (provider, model)
        self._llm_cache: Dict[tuple, Any] = {}
        # Models we've given up on this session (daily quota exhausted)
        self._exhausted: Set[tuple] = set()
        self._chain: List[Dict] = RAGConfig.LLM_FALLBACK_CHAIN
        # Tracks the model name used for the most recent successful call.
        # Read by orchestrator.reason_file() to populate FSMStore.reasoning_version.
        self._last_model_used: str = ""
        logger.info(
            "LLM fallback chain: %s",
            " → ".join(f"{e['provider']}:{e['model']}" for e in self._chain),
        )

    # ── Public API ────────────────────────────────────────────────────────

    def call(self, system_prompt: str, user_prompt: str, run_name: str = "llm_call") -> str:
        """Try each model in the fallback chain; return first successful response.

        Args:
            run_name: Label shown in LangSmith trace UI (e.g. 'entity_and_tag',
                      'backlink_batch').  Has no effect if tracing is disabled.
        """
        # Proactively throttle to stay within the API rate limit.
        # This blocks the calling thread; it is called from a scheduler
        # background thread so the watcher is unaffected.
        if _rate_limiter is not None:
            _rate_limiter.acquire()

        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_core.runnables import RunnableConfig

        for entry in self._chain:
            key = (entry["provider"], entry["model"])
            if key in self._exhausted:
                continue

            # Gemma models don't support SystemMessage — fold into HumanMessage
            if "gemma" in entry["model"].lower():
                messages = [HumanMessage(content=f"{system_prompt}\n\n{user_prompt}")]
            else:
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]

            llm = self._get_llm(entry)
            result = self._try_model(llm, messages, key, run_name=run_name)
            if result is not None:
                self._last_model_used = entry["model"]
                return result

        logger.error("All models in fallback chain exhausted or failed.")
        return ""

    def call_json(self, system_prompt: str, user_prompt: str, run_name: str = "llm_call") -> Optional[Dict[str, Any]]:
        """Like call() but automatically parses a JSON *object* from the response."""
        raw = self.call(system_prompt, user_prompt, run_name=run_name)
        return self._extract_json(raw)

    def call_json_list(self, system_prompt: str, user_prompt: str, run_name: str = "llm_call") -> List[Dict[str, Any]]:
        """Like call() but parses a JSON *array* from the response."""
        raw = self.call(system_prompt, user_prompt, run_name=run_name)
        return self.extract_json_array(raw)

    # ── Internal ──────────────────────────────────────────────────────────

    def _try_model(
        self,
        llm: Any,
        messages: list,
        key: tuple,
        run_name: str = "llm_call",
    ) -> Optional[str]:
        """
        Attempt up to LLM_MAX_RETRIES calls on this model.
        Returns the response string, or None if we should cascade.
        """
        from langchain_core.runnables import RunnableConfig
        provider, model = key
        cfg = RunnableConfig(run_name=run_name, tags=[provider, model])
        for attempt in range(1, RAGConfig.LLM_MAX_RETRIES + 1):
            try:
                response = llm.invoke(messages, cfg)
                return response.content.strip()
            except Exception as exc:
                exc_str = str(exc)
                retry_delay = self._parse_retry_delay(exc_str)
                is_rate_limit = retry_delay is not None
                is_daily_exhausted = (
                    "PerDay" in exc_str
                    or "quota" in exc_str.lower()
                    or "NOT_FOUND" in exc_str
                    or "404" in exc_str
                    or "INVALID_ARGUMENT" in exc_str
                    or "400" in exc_str
                )

                if is_daily_exhausted:
                    # Daily quota — no point retrying, cascade immediately
                    logger.warning(
                        "Daily quota exhausted for %s:%s — cascading to next model.",
                        provider, model,
                    )
                    self._exhausted.add(key)
                    return None

                if is_rate_limit and attempt < RAGConfig.LLM_MAX_RETRIES:
                    # RPM limit — wait the requested delay then retry same model
                    wait = max(retry_delay, RAGConfig.LLM_RETRY_BASE_SEC)
                    logger.warning(
                        "RPM limit on %s:%s (attempt %d/%d) — waiting %.0fs…",
                        provider, model, attempt, RAGConfig.LLM_MAX_RETRIES, wait,
                    )
                    time.sleep(wait)
                    continue

                # Non-rate-limit error or retries used up — cascade
                logger.error("Model %s:%s failed: %s — cascading.", provider, model, exc)
                return None

        return None  # retries exhausted

    def _get_llm(self, entry: Dict) -> Any:
        """Build (or return cached) LangChain LLM for this chain entry."""
        key = (entry["provider"], entry["model"])
        if key in self._llm_cache:
            return self._llm_cache[key]

        provider = entry["provider"]
        model    = entry["model"]
        logger.info("Initialising LLM: %s:%s", provider, model)

        if provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=RAGConfig.LLM_API_KEY,
                temperature=RAGConfig.LLM_TEMPERATURE,
                max_output_tokens=RAGConfig.LLM_MAX_TOKENS,
            )
        elif provider == "openai":
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=model,
                api_key=RAGConfig.LLM_API_KEY,
                temperature=RAGConfig.LLM_TEMPERATURE,
                max_tokens=RAGConfig.LLM_MAX_TOKENS,
            )
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(
                model=model,
                api_key=RAGConfig.LLM_API_KEY,
                temperature=RAGConfig.LLM_TEMPERATURE,
                max_tokens=RAGConfig.LLM_MAX_TOKENS,
            )
        elif provider == "ollama":
            from langchain_ollama import ChatOllama
            llm = ChatOllama(
                model=model,
                temperature=RAGConfig.LLM_TEMPERATURE,
            )
        else:
            raise ValueError(f"Unknown LLM provider: {provider!r}")

        self._llm_cache[key] = llm
        return llm

    @staticmethod
    def _parse_retry_delay(exc_str: str) -> Optional[float]:
        """Extract retryDelay seconds from a 429 error string, or return None."""
        if "429" not in exc_str and "RESOURCE_EXHAUSTED" not in exc_str:
            return None
        match = re.search(r"retryDelay[^:]*:\s*['\"]?(\d+)", exc_str)
        return float(match.group(1)) + 2.0 if match else RAGConfig.LLM_RETRY_BASE_SEC

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON object from possibly-markdown-wrapped LLM output."""
        text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError as e:
                logger.warning("JSON parse error: %s | raw: %s...", e, text[:200])
        return None

    @staticmethod
    def extract_json_array(text: str) -> List[Dict[str, Any]]:
        """Extract a JSON array from possibly-markdown-wrapped LLM output."""
        text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(0))
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError as e:
                logger.warning("JSON array parse error: %s | raw: %s...", e, text[:200])
        return []


# Module-level singleton
_llm_instance: Optional[LLMInterface] = None


def get_llm() -> LLMInterface:
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMInterface()
    return _llm_instance
