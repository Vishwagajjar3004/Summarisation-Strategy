"""
long_conversation_manager
==========================
Keeps a chat history under a token budget by automatically compressing
the oldest half of the conversation into a single summary message once
a threshold is crossed.

Design
------
- TokenCounter: approximates Claude's tokenizer using tiktoken's
  cl100k_base encoding. It's not byte-identical to Claude's real
  tokenizer, but it's within a few percent for English text and is
  the standard stand-in when the real tokenizer isn't available
  locally. Swap in `anthropic.Anthropic().messages.count_tokens(...)`
  for exact counts if you have API access.

- Summarizer: pluggable. `default_summarizer` is a small rule-based
  extractor (no external API calls) so this file runs standalone.
  `make_claude_summarizer()` shows how to swap in a real Claude call
  for production-quality summaries -- just pass it as `summarizer=`.

- ConversationManager: owns the message list, checks token totals
  after every turn, and folds the oldest 50% of messages into one
  summary message when the budget is exceeded. The summary is always
  kept as message[0] going forward, and future summarizations fold
  the *previous* summary + next batch of old messages together, so
  facts don't get lost after repeated compressions.
"""

from __future__ import annotations
import re
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

# NOTE on tokenization: ideally we'd count tokens with the exact tokenizer
# for the model in use (e.g. tiktoken's cl100k_base, or Anthropic's own
# `client.messages.count_tokens(...)` if you have API access). This sandbox
# can't reach the file tiktoken needs to download, so we fall back to the
# standard approximation used across the industry when an exact tokenizer
# isn't available: ~4 characters per token for English text, with a small
# correction for whitespace-delimited word count. This is within roughly
# 10-15% of real BPE tokenizers for ordinary prose, which is more than
# accurate enough for a compression *threshold* decision.


class TokenCounter:
    @staticmethod
    def count(text: str) -> int:
        if not text:
            return 0
        by_chars = len(text) / 4.0
        by_words = len(text.split()) / 0.75   # ~0.75 words per token, inverted
        return round((by_chars + by_words) / 2)

    @staticmethod
    def count_messages(messages: "List[Message]") -> int:
        # +4 tokens/message overhead approximates role/formatting tokens,
        # mirroring the convention OpenAI/Anthropic-style counters use.
        return sum(TokenCounter.count(m.content) + 4 for m in messages)


@dataclass
class Message:
    role: str          # "user" | "assistant" | "system"
    content: str
    ts: float = field(default_factory=time.time)
    is_summary: bool = False


# ---------------------------------------------------------------------
# Summarizers
# ---------------------------------------------------------------------

_FACT_KEYWORDS = [
    "my name is", "i'm ", "i am ",
    "i live in", "i work in", "i work as", "based in",
    "deadline is", "due date is", "budget is",
    "favorite", "favourite",
    "prefer", "i like", "i need",
]


def _split_sentences(text: str) -> List[str]:
    # simple sentence splitter -- good enough for chat-style text
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def default_summarizer(messages: List[Message]) -> str:
    """
    Rule-based extractive summarizer (no external API calls).
    Pulls out whole sentences that look like durable facts (names,
    locations, dates, preferences, numbers) plus a short topic recap,
    and discards small talk / filler. This keeps the demo self-contained;
    swap for make_claude_summarizer() in production for better prose
    quality on the topic-recap portion (fact extraction stays reliable
    either way since it's pattern-based, not generative).
    """
    facts = []
    topics = []
    for m in messages:
        if m.role != "user":
            continue
        for sentence in _split_sentences(m.content):
            low = sentence.lower()
            if any(k in low for k in _FACT_KEYWORDS):
                facts.append(sentence)
        words = m.content.split()
        if len(words) > 6:
            topics.append(" ".join(words[:6]) + "...")

    facts = list(dict.fromkeys(facts))       # de-dupe, keep order
    topics = list(dict.fromkeys(topics))[:5]

    lines = [f"[Summary of {len(messages)} earlier messages]"]
    if facts:
        lines.append("Key facts to remember:")
        lines += [f"- {f}" for f in facts]
    if topics:
        lines.append("Topics discussed: " + "; ".join(topics))
    return "\n".join(lines)


def make_claude_summarizer(api_key: Optional[str] = None, model: str = "claude-sonnet-4-6"):
    """
    Returns a summarizer function that calls the real Anthropic API.
    Not used by default (keeps this script runnable offline), but this
    is the production path -- pass it to ConversationManager(summarizer=...).
    """
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    def _summarize(messages: List[Message]) -> str:
        transcript = "\n".join(f"{m.role}: {m.content}" for m in messages)
        resp = client.messages.create(
            model=model,
            max_tokens=400,
            messages=[{
                "role": "user",
                "content": (
                    "Summarize this conversation excerpt into a concise "
                    "paragraph. Preserve every concrete fact (names, dates, "
                    "numbers, preferences, decisions) exactly. Omit small talk.\n\n"
                    f"{transcript}"
                ),
            }],
        )
        return "".join(b.text for b in resp.content if b.type == "text")

    return _summarize


# ---------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------

class ConversationManager:
    def __init__(
        self,
        token_limit: int = 3000,
        summarizer: Optional[Callable[[List[Message]], str]] = None,
    ):
        self.token_limit = token_limit
        self.summarizer = summarizer or default_summarizer
        self.messages: List[Message] = []
        self.summary: Optional[Message] = None
        self.compressions: List[dict] = []   # audit trail for the test

    # -- public API ------------------------------------------------
    def add_message(self, role: str, content: str) -> None:
        self.messages.append(Message(role=role, content=content))
        self._maybe_condense()

    def get_context(self) -> List[Message]:
        """What you'd actually send to the model this turn."""
        return ([self.summary] if self.summary else []) + self.messages

    def total_tokens(self) -> int:
        return TokenCounter.count_messages(self.get_context())

    def raw_total_tokens(self) -> int:
        """Token count if nothing had ever been summarized (for comparison)."""
        all_msgs = self.messages if not self.summary else \
            [m for c in self.compressions for m in c["folded_messages"]] + self.messages
        return TokenCounter.count_messages(all_msgs)

    # -- internals ---------------------------------------------------
    def _maybe_condense(self) -> None:
        if self.total_tokens() <= self.token_limit:
            return

        pool = ([self.summary] if self.summary else []) + self.messages
        cut = max(1, len(pool) // 2)          # oldest 50%
        old_half, recent_half = pool[:cut], pool[cut:]

        summary_text = self.summarizer(old_half)
        self.compressions.append({
            "at_message_count": len(self.messages),
            "folded_messages": [m for m in old_half if not m.is_summary],
            "summary_text": summary_text,
        })

        self.summary = Message(role="system", content=summary_text, is_summary=True)
        # recent_half may itself still contain the old summary marker; strip it
        self.messages = [m for m in recent_half if not m.is_summary]

    # -- debugging -----------------------------------------------------
    def describe(self) -> str:
        ctx = self.get_context()
        lines = [f"context messages: {len(ctx)}  |  tokens: {self.total_tokens()}"]
        if self.summary:
            lines.append(f"  [SUMMARY] ({TokenCounter.count(self.summary.content)} tok): "
                          f"{self.summary.content[:120]}...")
        for m in self.messages[:3]:
            lines.append(f"  ...{m.role}: {m.content[:60]}")
        return "\n".join(lines)
