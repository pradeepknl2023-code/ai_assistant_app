"""
ai_router.py  ·  Gemini-Only Router  ·  v3.0
=============================================
Stripped down to ONLY Google Gemini FREE models.
No LiteLLM complexity. Uses litellm under the hood (same as before)
but only two Gemini configs — no Groq, no OpenAI, no Anthropic, no Ollama.

Provider priority:
  Tier 1 — gemini/gemini-2.0-flash   (PRIMARY,  FREE, 1M TPM)
  Tier 2 — gemini/gemini-1.5-flash   (FALLBACK, FREE, 1M TPM)
    → Auto-used when 2.0-flash is rate-limited. No manual intervention needed.

How to configure:
  Set GEMINI_API_KEY in Streamlit Secrets (or as an environment variable):
    [secrets]
    GEMINI_API_KEY = "AIzaSy..."
  Get a free key at: https://aistudio.google.com/apikey

app.py interface — UNCHANGED (drop-in replacement):
  call_ai_compat(), get_router_status(), get_active_provider(),
  get_active_model(), get_next_provider(), reset_cooldowns(),
  _get_key(), RATE_LIMIT_SENTINEL
"""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger("AI_ROUTER")

# ─────────────────────────────────────────────────────────────
# GLOBAL LAST-USED TRACKING  (same as original)
# ─────────────────────────────────────────────────────────────
_LAST_USED_PROVIDER = "None yet"
_LAST_USED_MODEL    = "—"

# Sentinel returned to app.py when all providers are exhausted
RATE_LIMIT_SENTINEL = "__RATE_LIMIT__"


# ─────────────────────────────────────────────────────────────
# KEY LOOKUP — os.environ first, then st.secrets  (unchanged)
# ─────────────────────────────────────────────────────────────
def _get_key(env_key: str) -> str:
    if not env_key:
        return ""
    val = os.environ.get(env_key, "").strip()
    if val:
        return val
    try:
        import streamlit as st
        val = str(st.secrets.get(env_key, "")).strip()
        if val:
            os.environ[env_key] = val
            logger.info(f"[ROUTER] {env_key} loaded from st.secrets")
        return val
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────
# MODEL REGISTRY — Gemini only
# ─────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    model: str
    env_key: str
    tpm: int
    rpd: int
    quality: int
    cost_per_1k: float
    provider: str
    task_types: list = field(default_factory=lambda: ["code", "jira", "summary"])


# ONLY Gemini models — primary + fallback
ALL_MODELS: list[ModelConfig] = [
    # PRIMARY — Gemini 2.0 Flash
    ModelConfig(
        model        = "gemini/gemini-2.0-flash",
        env_key      = "GEMINI_API_KEY",
        tpm          = 1_000_000,
        rpd          = 1500,
        quality      = 5,
        cost_per_1k  = 0.0,
        provider     = "Google Gemini 2.0 Flash",
    ),
    # FALLBACK — Gemini 1.5 Flash (auto-used when 2.0 is rate-limited)
    ModelConfig(
        model        = "gemini/gemini-1.5-flash",
        env_key      = "GEMINI_API_KEY",
        tpm          = 1_000_000,
        rpd          = 1500,
        quality      = 4,
        cost_per_1k  = 0.0,
        provider     = "Google Gemini 1.5 Flash",
    ),
]

# Token budgets per task type
TASK_TOKENS = {
    "code"   : 1200,
    "summary": 350,
    "jira"   : 4000,
}


# ─────────────────────────────────────────────────────────────
# PER-MODEL RUNTIME STATUS
# ─────────────────────────────────────────────────────────────
@dataclass
class _ModelStatus:
    rate_limited_until: float = 0.0
    auth_failed       : bool  = False
    model_gone        : bool  = False
    total_calls       : int   = 0
    total_tokens      : int   = 0
    last_used_ts      : float = 0.0
    rate_limit_count  : int   = 0


_status_registry: dict[str, _ModelStatus] = {}


def _st(model: str) -> _ModelStatus:
    if model not in _status_registry:
        _status_registry[model] = _ModelStatus()
    return _status_registry[model]


def reset_cooldowns() -> int:
    """
    Clear all cooldowns, model_gone flags, and auth_failed flags.
    Called by the '⚡ Reset All Cooldowns' button in the UI.
    Returns the number of providers recovered.
    """
    cleared = 0
    for model, s in _status_registry.items():
        changed = False
        if s.rate_limited_until > time.time():
            s.rate_limited_until = 0.0
            s.rate_limit_count   = 0
            changed = True
        if s.model_gone:
            s.model_gone = False
            changed = True
        if s.auth_failed:
            s.auth_failed = False
            changed = True
        if changed:
            cleared += 1
    logger.info(f"[ROUTER] reset_cooldowns — {cleared} provider(s) recovered")
    return cleared


# ─────────────────────────────────────────────────────────────
# AVAILABILITY CHECK
# ─────────────────────────────────────────────────────────────
def _is_available(cfg: ModelConfig) -> bool:
    s = _st(cfg.model)
    if s.auth_failed or s.model_gone:
        return False
    if time.time() < s.rate_limited_until:
        return False
    # Key must be present
    if not _get_key(cfg.env_key):
        return False
    return True


def _mark_rate_limit(model: str):
    """Graduated back-off: 30 s → 60 s → 90 s (cap)."""
    s = _st(model)
    s.rate_limit_count  += 1
    backoff              = min(30 * s.rate_limit_count, 90)
    s.rate_limited_until = time.time() + backoff
    logger.warning(f"[ROUTER] {model} cooldown={backoff}s (hit #{s.rate_limit_count})")


def _mark_success(model: str, tokens: int = 0, provider: str = "", display_model: str = ""):
    global _LAST_USED_PROVIDER, _LAST_USED_MODEL
    s = _st(model)
    s.total_calls       += 1
    s.total_tokens      += tokens
    s.last_used_ts       = time.time()
    s.rate_limited_until = 0.0
    s.rate_limit_count   = 0
    if provider:
        _LAST_USED_PROVIDER = provider
    if display_model:
        _LAST_USED_MODEL = display_model


# ─────────────────────────────────────────────────────────────
# ERROR CLASSIFIERS
# ─────────────────────────────────────────────────────────────
def _is_rate_err(exc: Exception) -> bool:
    s = (str(type(exc).__name__) + str(exc)).lower()
    return any(x in s for x in [
        "rate", "429", "ratelimit", "quota", "tokens per",
        "requests per", "toomanyrequests", "resource_exhausted",
        "too many requests",
    ])


def _is_auth_err(exc: Exception) -> bool:
    s = str(exc).lower()
    return any(x in s for x in [
        "401", "403", "invalid_api_key", "authentication",
        "unauthorized", "permission_denied", "api key not valid",
    ])


def _is_gone_err(exc: Exception) -> bool:
    s = str(exc).lower()
    return any(x in s for x in [
        "decommissioned", "not supported", "no longer supported",
        "model_not_found", "does not exist", "invalid argument",
        "unsupported",
    ])


# ─────────────────────────────────────────────────────────────
# CORE ROUTER — Gemini 2.0 Flash → 1.5 Flash fallback
# ─────────────────────────────────────────────────────────────
def call_ai(
    messages     : list[dict],
    temperature  : float = 0.1,
    max_tokens   : int   = 1200,
    task         : Literal["code", "summary", "jira"] = "code",
    require_free : bool  = False,   # kept for API compatibility, always True here
    force_provider: str | None = None,
) -> str:
    """
    Call the best available Gemini model.
    Returns the response string, or RATE_LIMIT_SENTINEL if all models are exhausted.
    """
    try:
        import litellm
        litellm.drop_params  = True
        litellm.set_verbose  = False
    except ImportError:
        raise RuntimeError(
            "litellm not installed.\n"
            "Run:  pip install litellm\n"
            "Then restart the app."
        )

    token_limit = TASK_TOKENS.get(task, max_tokens)
    api_key     = _get_key("GEMINI_API_KEY")

    if not api_key:
        logger.error("[ROUTER] GEMINI_API_KEY is not set.")
        return RATE_LIMIT_SENTINEL

    # Build candidate list — both Gemini models, filtered by availability
    candidates = [
        cfg for cfg in ALL_MODELS
        if task in cfg.task_types
        and _is_available(cfg)
        and (not force_provider or cfg.provider == force_provider)
    ]

    if not candidates:
        logger.error("[ROUTER] No Gemini models available (both rate-limited or auth failed).")
        return RATE_LIMIT_SENTINEL

    logger.info(f"[ROUTER] Candidates for task={task}: {[c.provider for c in candidates]}")

    for cfg in candidates:
        for attempt in range(2):            # 1 retry per model on rate-limit
            try:
                logger.info(
                    f"[ROUTER] → {cfg.provider} [{cfg.model}] "
                    f"attempt={attempt+1} task={task} max_tokens={token_limit}"
                )
                resp = litellm.completion(
                    model       = cfg.model,
                    messages    = messages,
                    temperature = temperature,
                    max_tokens  = token_limit,
                    api_key     = api_key,
                )
                content = resp.choices[0].message.content or ""
                tokens  = getattr(resp.usage, "total_tokens", 0)

                _mark_success(
                    cfg.model,
                    tokens,
                    provider     = cfg.provider,
                    display_model= cfg.model.split("/")[-1],
                )
                logger.info(
                    f"[ROUTER] SUCCESS: {cfg.provider} | {tokens} tokens | task={task}"
                )
                return content

            except Exception as exc:
                logger.warning(f"[ROUTER] {cfg.provider} error: {exc}")

                if _is_auth_err(exc):
                    _st(cfg.model).auth_failed = True
                    logger.error(
                        "[ROUTER] Auth failed — check GEMINI_API_KEY in Streamlit Secrets.\n"
                        "         Get a free key at: https://aistudio.google.com/apikey"
                    )
                    break   # skip to next model (1.5-flash won't help if key is wrong)

                if _is_gone_err(exc):
                    _st(cfg.model).model_gone = True
                    logger.warning(f"[ROUTER] {cfg.model} unavailable/unsupported, trying fallback.")
                    break

                if _is_rate_err(exc):
                    if attempt == 0:
                        logger.info("[ROUTER] Rate limit on attempt 1 — retrying in 3 s")
                        time.sleep(3)
                        continue
                    # Second attempt also rate-limited → cool down this model
                    _mark_rate_limit(cfg.model)
                    logger.info(
                        f"[ROUTER] {cfg.provider} rate-limited, "
                        f"{'trying Gemini 1.5 Flash fallback' if cfg.model.endswith('2.0-flash') else 'all models exhausted'}."
                    )
                    break   # fall through to next candidate (1.5-flash)

                # Unknown error — don't retry, try next model
                logger.warning("[ROUTER] Unknown error — skipping to fallback.")
                break

    logger.error("[ROUTER] All Gemini models exhausted.")
    return RATE_LIMIT_SENTINEL


# ─────────────────────────────────────────────────────────────
# BACKWARD-COMPATIBLE SHIM  (app.py imports this as call_ai)
# ─────────────────────────────────────────────────────────────
def call_ai_compat(
    messages    : list,
    temperature : float = 0.1,
    model       : str   = None,   # ignored — Gemini chosen automatically
    task        : str   = "code",
) -> str:
    return call_ai(messages, temperature=temperature, task=task)


# ─────────────────────────────────────────────────────────────
# UI HELPERS  (same signatures as original — app.py calls these)
# ─────────────────────────────────────────────────────────────
def get_router_status() -> list[dict]:
    """Returns a list of dicts for st.dataframe() in the UI."""
    rows = []
    has_key = bool(_get_key("GEMINI_API_KEY"))

    for cfg in ALL_MODELS:
        s   = _st(cfg.model)
        now = time.time()

        if not has_key:
            status = "⚪ No API Key"
        elif s.auth_failed:
            status = "🔴 Auth Failed — check GEMINI_API_KEY"
        elif s.model_gone:
            status = "⚫ Model Unavailable"
        elif now < s.rate_limited_until:
            remaining = int(s.rate_limited_until - now)
            status = f"🟡 Cooldown {remaining}s (hit #{s.rate_limit_count})"
        else:
            status = "🟢 Ready"

        is_last = (cfg.provider == _LAST_USED_PROVIDER)
        label   = ("⭐ LAST USED → " if is_last else "") + cfg.provider

        rows.append({
            "Provider" : label,
            "Model"    : cfg.model.split("/")[-1],
            "Cost"     : "FREE",
            "TPM"      : f"{cfg.tpm:,}",
            "Quality"  : "★" * cfg.quality + "☆" * (5 - cfg.quality),
            "Status"   : status,
            "Calls"    : s.total_calls,
            "Tokens"   : f"{s.total_tokens:,}",
        })
    return rows


def get_active_provider(task: str = "code") -> str:
    """Returns the last successfully used provider, or the next ready one."""
    if _LAST_USED_PROVIDER != "None yet":
        return _LAST_USED_PROVIDER
    for cfg in ALL_MODELS:
        if _is_available(cfg) and task in cfg.task_types:
            return cfg.provider
    return "No Gemini key configured"


def get_active_model(task: str = "code") -> str:
    """Returns the last used model name, or the next ready one."""
    if _LAST_USED_MODEL != "—":
        return _LAST_USED_MODEL
    for cfg in ALL_MODELS:
        if _is_available(cfg) and task in cfg.task_types:
            return cfg.model.split("/")[-1]
    return "—"


def get_next_provider(task: str = "code") -> str:
    """Returns the provider that will be used on the next call."""
    for cfg in ALL_MODELS:
        if _is_available(cfg) and task in cfg.task_types:
            return cfg.provider
    return "None available — check GEMINI_API_KEY"
