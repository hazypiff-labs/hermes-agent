"""Observation-masking context engine (JetBrains LastNObservations).

Reference implementation:
  https://github.com/JetBrains-Research/the-complexity-trap
  paper: https://arxiv.org/abs/2508.21433

The claim under test is that simply MASKING old tool observations matches or
beats LLM summarization at a fraction of the cost. This engine exists to let the
replay suite measure whether that transfers to our harnesses -- it is not assumed.

Design, following the reference:

  * Mask ONLY observations (role="tool"). Assistant reasoning and tool CALLS are
    preserved verbatim -- the agent must still see what it did and why, or it
    repeats work. This is the single most important property; dropping actions
    is what makes naive truncation fail.
  * The most recent ``keep_last_n`` observations stay intact.
  * Masked content is replaced by a MECHANICAL placeholder (line/char counts).
    No LLM call, so masking is free and cannot hallucinate.
  * ``keep_output`` / ``remove_output`` exceptions: some tools are never worth
    masking (tiny results) and some are always worth masking (huge dumps).
  * POLLING: the mask boundary only advances every ``poll_every`` turns. Moving
    it every turn would rewrite the prompt prefix on every request and destroy
    prompt-cache reuse -- which would cost more than masking saves.

Falls back to the built-in ContextCompressor when masking alone cannot fit the
budget, so this is strictly additive: masking first, summarize only if needed.

Selection: ``context.engine: masking`` in config.yaml.
Set HERMES_CONTEXT_MODE=current to disable masking at runtime (A/B control)
without editing config -- the replay suite relies on this.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

# Aliased deliberately. The loader's fallback scans dir(module) for any
# ContextEngine subclass and instantiates the first it finds; importing the base
# under its own name risks the loader returning the BUILT-IN compressor instead of
# this engine, which would make an A/B silently compare "current" against
# "current". register() below takes priority, and the alias is belt-and-braces.
try:
    from agent.context_compressor import ContextCompressor as _BaseCompressor
except Exception:  # pragma: no cover
    _BaseCompressor = None

ContextCompressor = _BaseCompressor


def _load_model_cfg() -> Dict[str, Any]:
    """Read the model block from config.yaml so the engine can be constructed
    with no arguments (the plugin loader instantiates it bare)."""
    try:
        import yaml
        from pathlib import Path
        p = Path(os.path.expanduser("~/.hermes/config.yaml"))
        with open(p, encoding="utf-8-sig") as f:
            return (yaml.safe_load(f) or {}).get("model", {}) or {}
    except Exception:
        return {}


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


# Tools whose output is small/structural enough that masking buys nothing.
DEFAULT_KEEP_OUTPUT = {"todo_write", "task_update", "memory_write"}
# Tools that reliably emit large dumps worth masking aggressively.
DEFAULT_REMOVE_OUTPUT = {"read_file", "bash", "shell", "grep", "search", "web_fetch"}

PLACEHOLDER = "[observation elided: {lines} lines, {chars} chars, tool={tool}]"


class MaskingContextEngine(ContextCompressor if ContextCompressor else object):
    """LastNObservations masking layered over the built-in compressor."""

    @property
    def name(self) -> str:
        return "masking"

    def __init__(self, model: str = "", **kw):
        if _BaseCompressor:
            cfg = _load_model_cfg()
            kw.setdefault("base_url", cfg.get("base_url", ""))
            kw.setdefault("api_key", cfg.get("api_key", ""))
            kw.setdefault("provider", cfg.get("provider", ""))
            kw.setdefault("config_context_length", cfg.get("context_length"))
            super().__init__(model or cfg.get("default", ""), **kw)
        self.keep_last_n = _int_env("HERMES_MASK_KEEP_LAST_N", 5)
        self.poll_every = _int_env("HERMES_MASK_POLL_EVERY", 3)
        self.min_mask_chars = _int_env("HERMES_MASK_MIN_CHARS", 200)
        self.keep_output = set(DEFAULT_KEEP_OUTPUT)
        self.remove_output = set(DEFAULT_REMOVE_OUTPUT)
        self._turn = 0
        self._masked_boundary = 0     # index below which observations are masked
        self.mask_count = 0
        self.masked_chars = 0

    # -- runtime A/B switch ------------------------------------------------

    @property
    def masking_enabled(self) -> bool:
        return os.environ.get("HERMES_CONTEXT_MODE", "masking").lower() != "current"

    def is_available(self) -> bool:
        return ContextCompressor is not None

    # -- masking core ------------------------------------------------------

    @staticmethod
    def _tool_name(msg: Dict[str, Any]) -> str:
        return (msg.get("name") or msg.get("tool_name")
                or (msg.get("tool_call_id") or "")[:24] or "unknown")

    def _should_mask(self, msg: Dict[str, Any]) -> bool:
        if msg.get("_masked"):
            return False
        content = msg.get("content")
        if not isinstance(content, str) or len(content) < self.min_mask_chars:
            return False
        tool = self._tool_name(msg).lower()
        if any(k in tool for k in self.keep_output):
            return False
        return True

    def apply_masking(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Mask all but the last N observations. Pure, cheap, no LLM call."""
        obs_idx = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
        if len(obs_idx) <= self.keep_last_n:
            return messages

        cutoff = obs_idx[-self.keep_last_n] if self.keep_last_n else len(messages)
        # Polling: only advance the boundary, never retreat, and only every
        # poll_every turns. A boundary that moves each turn invalidates the
        # prompt cache on every request.
        if cutoff <= self._masked_boundary:
            cutoff = self._masked_boundary
        elif self._turn % self.poll_every != 0:
            cutoff = self._masked_boundary
        self._masked_boundary = cutoff

        out = []
        for i, m in enumerate(messages):
            if i < cutoff and m.get("role") == "tool" and self._should_mask(m):
                c = m["content"]
                new = dict(m)
                new["content"] = PLACEHOLDER.format(
                    lines=c.count("\n") + 1, chars=len(c),
                    tool=self._tool_name(m))
                new["_masked"] = True
                self.mask_count += 1
                self.masked_chars += len(c) - len(new["content"])
                out.append(new)
            else:
                out.append(m)
        return out

    # -- ContextEngine hooks ----------------------------------------------

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        self._turn += 1
        if ContextCompressor:
            super().update_from_response(usage)

    def should_compress(self, prompt_tokens: int = None) -> bool:
        # Masking runs whenever there is anything to mask; summarization still
        # only fires on the built-in threshold.
        if self.masking_enabled:
            return True
        return super().should_compress(prompt_tokens) if ContextCompressor else False

    def compress(self, messages, current_tokens=None, focus_topic=None):
        if not self.masking_enabled:
            return (super().compress(messages, current_tokens, focus_topic)
                    if ContextCompressor else messages)

        masked = self.apply_masking(messages)

        # Masking alone may not be enough near the ceiling -- fall back to real
        # summarization rather than letting the request overflow.
        if ContextCompressor and super().should_compress(current_tokens):
            return super().compress(masked, current_tokens, focus_topic)
        return masked

    def get_status(self) -> Dict[str, Any]:
        s = (super().get_status() if ContextCompressor else {})
        s.update(engine="masking", masking_enabled=self.masking_enabled,
                 masked_observations=self.mask_count,
                 masked_chars_saved=self.masked_chars,
                 mask_boundary=self._masked_boundary,
                 keep_last_n=self.keep_last_n, poll_every=self.poll_every)
        return s

    def on_session_reset(self) -> None:
        if ContextCompressor:
            super().on_session_reset()
        self._turn = 0
        self._masked_boundary = 0
        self.mask_count = 0
        self.masked_chars = 0


def register(ctx):
    """Preferred load path. Takes priority over the loader's dir()-scan fallback,
    which cannot construct either class (ContextCompressor requires `model`)."""
    ctx.register_context_engine(MaskingContextEngine())
