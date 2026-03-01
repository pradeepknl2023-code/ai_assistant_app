"""
ai_router.py  ·  Gemini-Only Router  ·  v3.1  (BUG-FIX RELEASE)
=================================================================
ROOT CAUSE FIX — Why "All AI Providers Rate-Limited" appeared on first call:

  BUG 1 — UNCLASSIFIED ERRORS were silently exhausting both models.
           Errors like ConnectionError, 500, 503, BadRequestError,
           "Invalid API key" (exact wording variant) did not match
           ANY of the three classifiers (_is_rate_err / _is_auth_err /
           _is_gone_err). The code fell through to the generic
           "Unknown error — skip to fallback" branch.
           When 1.5-flash also failed the same way, the loop exhausted
           and returned RATE_LIMIT_SENTINEL.
           The UI then showed "All AI Providers Rate-Limited" — wrong.
           The user had NOT hit any rate limit at all.

  BUG 2 — _is_gone_err missed "not found" (without underscore).
           Gemini returns: "model: gemini/gemini-2.0-flash is not found"
           The pattern list had "model_not_found" but not "not found",
           so this slipped through to the unclassified path.

  BUG 3 — _is_auth_err missed "invalid api key" (exact phrase from Google).
           Google returns: "Invalid API key. Please pass a valid API key."
           The pattern list had "invalid_api_key" (with underscore, litellm
           internal name) but not "invalid api key" (Google's actual message).
           Result: auth errors were also unclassified → wrong error shown.

  BUG 4 — Transient errors (500, 503, network timeouts) were treated the
           same as permanent failures. They should retry after a short wait,
           not immediately exhaust the model.

  BUG 5 — The last_error was never surfaced to the caller. app.py only
           received RATE_LIMIT_SENTINEL with no explanation, so the UI
           always showed the generic rate-limit message regardless of
           the actual error.

  v3.1 FIXES:
  ──────────────────────────────────────────────────────────────
  • _is_auth_err:      added "invalid api key", "api_key_invalid", "key not valid"
  • _is_gone_err:      added "not found", "notfound", "does not exist"
  • _is_transient_err: NEW — catches 500, 503, connection errors, timeouts
                       → short sleep + retry instead of model exhaustion
  • _is_bad_request:   NEW — catches 400 / BadRequest (bad model string, etc.)
                       → marks model_gone so fallback is tried
  • last_error global: stores the real exception string so app.py can show it
  • Logging improved:  every branch logs clearly what happened and why

Provider priority (unchanged):
  Tier 1 — gemini/gemini-2.0-flash   (PRIMARY,  FREE, 1M TPM)
  Tier 2 — gemini/gemini-1.5-flash   (FALLBACK, FREE, 1M TPM)
"""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger("AI_ROUTER")

# ─────────────────────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────────────────────
_LAST_USED_PROVIDER = "None yet"
_LAST_USED_MODEL    = "—"
_LAST_ERROR         = ""          # ← NEW: real error exposed to app.py / UI

RATE_LIMIT_SENTINEL = "__RATE_LIMIT__"


# ─────────────────────────────────────────────────────────────
# KEY LOOKUP — os.environ first, then st.secrets
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


def get_last_error() -> str:
    """Returns the real error message from the last failed call."""
    return _LAST_ERROR


# ─────────────────────────────────────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    model       : str
    env_key     : str
    tpm         : int
    rpd         : int
    quality     : int
    cost_per_1k : float
    provider    : str
    task_types  : list = field(default_factory=lambda: ["code", "jira", "summary"])


ALL_MODELS: list[ModelConfig] = [
    # PRIMARY — Gemini 2.0 Flash
    ModelConfig(
        model       = "gemini/gemini-2.0-flash",
        env_key     = "GEMINI_API_KEY",
        tpm         = 1_000_000,
        rpd         = 1500,
        quality     = 5,
        cost_per_1k = 0.0,
        provider    = "Google Gemini 2.0 Flash",
    ),
    # FALLBACK — Gemini 1.5 Flash (auto-used when 2.0 is rate-limited)
    ModelConfig(
        model       = "gemini/gemini-1.5-flash",
        env_key     = "GEMINI_API_KEY",
        tpm         = 1_000_000,
        rpd         = 1500,
        quality     = 4,
        cost_per_1k = 0.0,
        provider    = "Google Gemini 1.5 Flash",
    ),
]

TASK_TOKENS = {"code": 1200, "summary": 350, "jira": 4000}


# ─────────────────────────────────────────────────────────────
# PER-MODEL RUNTIME STATUS
# ─────────────────────────────────────────────────────────────
@dataclass
class _ModelStatus:
    rate_limited_until : float = 0.0
    auth_failed        : bool  = False
    model_gone         : bool  = False
    total_calls        : int   = 0
    total_tokens       : int   = 0
    last_used_ts       : float = 0.0
    rate_limit_count   : int   = 0


_status_registry: dict[str, _ModelStatus] = {}


def _st(model: str) -> _ModelStatus:
    if model not in _status_registry:
        _status_registry[model] = _ModelStatus()
    return _status_registry[model]


def reset_cooldowns() -> int:
    """Clear all cooldowns / model_gone / auth_failed. Returns count cleared."""
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
# AVAILABILITY
# ─────────────────────────────────────────────────────────────
def _is_available(cfg: ModelConfig) -> bool:
    s = _st(cfg.model)
    if s.auth_failed or s.model_gone:
        return False
    if time.time() < s.rate_limited_until:
        return False
    if not _get_key(cfg.env_key):
        return False
    return True


def _mark_rate_limit(model: str):
    """Graduated back-off: 30 s → 60 s → 90 s cap."""
    s = _st(model)
    s.rate_limit_count  += 1
    backoff              = min(30 * s.rate_limit_count, 90)
    s.rate_limited_until = time.time() + backoff
    logger.warning(f"[ROUTER] {model} rate-limit cooldown={backoff}s (hit #{s.rate_limit_count})")


def _mark_success(model: str, tokens: int = 0, provider: str = "", display_model: str = ""):
    global _LAST_USED_PROVIDER, _LAST_USED_MODEL, _LAST_ERROR
    s = _st(model)
    s.total_calls       += 1
    s.total_tokens      += tokens
    s.last_used_ts       = time.time()
    s.rate_limited_until = 0.0
    s.rate_limit_count   = 0
    _LAST_ERROR          = ""   # clear error on success
    if provider:
        _LAST_USED_PROVIDER = provider
    if display_model:
        _LAST_USED_MODEL = display_model


# ─────────────────────────────────────────────────────────────
# ERROR CLASSIFIERS  (v3.1 — expanded and corrected)
# ─────────────────────────────────────────────────────────────

def _is_rate_err(exc: Exception) -> bool:
    """True for genuine rate-limit / quota errors (HTTP 429)."""
    s = (str(type(exc).__name__) + " " + str(exc)).lower()
    return any(x in s for x in [
        "429",
        "rate limit",
        "ratelimit",
        "rate_limit",
        "quota",
        "tokens per",
        "requests per",
        "toomanyrequests",
        "resource_exhausted",
        "too many requests",
    ])


def _is_auth_err(exc: Exception) -> bool:
    """
    True for authentication / bad-key errors.
    FIX v3.1: added Google's actual error phrases:
      'invalid api key', 'key not valid', 'api_key_invalid'
    """
    s = (str(type(exc).__name__) + " " + str(exc)).lower()
    return any(x in s for x in [
        "401",
        "403",
        "invalid_api_key",          # litellm internal name
        "invalid api key",          # ← FIX: Google's actual phrase
        "api key not valid",        # ← FIX: Google's actual phrase
        "api_key_invalid",          # ← FIX: alternative spelling
        "key not valid",            # ← FIX: short form
        "authentication",
        "unauthorized",
        "permission_denied",
        "authenticationerror",
    ])


def _is_gone_err(exc: Exception) -> bool:
    """
    True for permanent model-unavailable errors.
    FIX v3.1: added 'not found', 'notfound' — Google returns
    'model: gemini/gemini-2.0-flash is not found' which the old
    pattern list (only 'model_not_found') was missing.
    """
    s = (str(type(exc).__name__) + " " + str(exc)).lower()
    return any(x in s for x in [
        "decommissioned",
        "not supported",
        "no longer supported",
        "model_not_found",
        "not found",               # ← FIX: catches Google's actual phrase
        "notfound",                # ← FIX: no-space variant
        "does not exist",
        "unsupported",
        # NOTE: "invalid argument" intentionally NOT here — lives in _is_bad_request
    ])


def _is_transient_err(exc: Exception) -> bool:
    """
    NEW v3.1 — True for temporary server/network errors worth retrying.
    These are NOT rate limits — they are infrastructure blips.
    500, 503, connection timeouts all belong here.
    """
    s = (str(type(exc).__name__) + " " + str(exc)).lower()
    return any(x in s for x in [
        "500",
        "503",
        "502",
        "504",
        "service unavailable",
        "serviceunavailable",
        "connection",
        "timeout",
        "timed out",
        "network",
        "temporarily unavailable",
        "internal server error",
        "max retries exceeded",
        "httpsconnectionpool",
    ])


def _is_bad_request(exc: Exception) -> bool:
    """
    NEW v3.1 — True for malformed-request errors (400 / BadRequest).
    Usually means bad model string or wrong parameter.
    We mark model_gone so the fallback is tried.
    """
    s = (str(type(exc).__name__) + " " + str(exc)).lower()
    return any(x in s for x in [
        "400",
        "bad request",
        "badrequest",
        "badrequesterror",
        "invalid_request",
        "invalidrequest",
    ])


# ─────────────────────────────────────────────────────────────
# CORE ROUTER
# ─────────────────────────────────────────────────────────────
def call_ai(
    messages      : list[dict],
    temperature   : float = 0.1,
    max_tokens    : int   = 1200,
    task          : Literal["code", "summary", "jira"] = "code",
    require_free  : bool  = False,
    force_provider: str | None = None,
) -> str:
    """
    Call the best available Gemini model.
    Returns response string, or RATE_LIMIT_SENTINEL if all models fail.
    The real error is always stored in _LAST_ERROR and logged.
    """
    global _LAST_ERROR

    try:
        import litellm
        litellm.drop_params = True
        litellm.set_verbose = False
    except ImportError:
        raise RuntimeError(
            "litellm not installed.\n"
            "Add 'litellm' to requirements.txt and redeploy."
        )

    token_limit = TASK_TOKENS.get(task, max_tokens)
    api_key     = _get_key("GEMINI_API_KEY")

    if not api_key:
        _LAST_ERROR = (
            "GEMINI_API_KEY is not set. "
            "Add it in Streamlit Cloud → Settings → Secrets:\n"
            "GEMINI_API_KEY = \"AIzaSy...\"\n"
            "Get a free key at https://aistudio.google.com/apikey"
        )
        logger.error(f"[ROUTER] {_LAST_ERROR}")
        return RATE_LIMIT_SENTINEL

    candidates = [
        cfg for cfg in ALL_MODELS
        if task in cfg.task_types
        and _is_available(cfg)
        and (not force_provider or cfg.provider == force_provider)
    ]

    if not candidates:
        _LAST_ERROR = (
            "All Gemini models are in cooldown or unavailable. "
            "Click ⚡ Reset All Cooldowns and try again, "
            "or check that GEMINI_API_KEY is valid."
        )
        logger.error(f"[ROUTER] {_LAST_ERROR}")
        return RATE_LIMIT_SENTINEL

    logger.info(f"[ROUTER] Candidates task={task}: {[c.provider for c in candidates]}")

    for cfg in candidates:
        for attempt in range(2):
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
                    cfg.model, tokens,
                    provider     = cfg.provider,
                    display_model= cfg.model.split("/")[-1],
                )
                logger.info(f"[ROUTER] SUCCESS: {cfg.provider} | {tokens} tokens | task={task}")
                return content

            except Exception as exc:
                exc_str    = str(exc)
                exc_detail = f"{type(exc).__name__}: {exc_str[:300]}"
                logger.warning(f"[ROUTER] {cfg.provider} error: {exc_detail}")
                _LAST_ERROR = exc_detail   # always store real error

                # ── AUTH error: bad/missing key ───────────────────
                if _is_auth_err(exc):
                    _st(cfg.model).auth_failed = True
                    _LAST_ERROR = (
                        f"Authentication failed for {cfg.provider}.\n"
                        f"Your GEMINI_API_KEY appears to be invalid or expired.\n"
                        f"Check Streamlit Cloud → Settings → Secrets.\n"
                        f"Get a new free key at https://aistudio.google.com/apikey\n"
                        f"Detail: {exc_str[:200]}"
                    )
                    logger.error(f"[ROUTER] AUTH FAIL — {_LAST_ERROR}")
                    break   # try 1.5-flash (same key, but log clearly)

                # ── GONE / BAD MODEL: model unavailable or bad request ─
                if _is_gone_err(exc) or _is_bad_request(exc):
                    _st(cfg.model).model_gone = True
                    reason = "gone/unsupported" if _is_gone_err(exc) else "bad request (model string issue)"
                    logger.warning(
                        f"[ROUTER] {cfg.provider} {reason} — "
                        f"{'trying 1.5-flash fallback' if '2.0' in cfg.model else 'no more fallback'}"
                    )
                    break   # try next candidate

                # ── RATE LIMIT: genuine 429 ───────────────────────
                if _is_rate_err(exc):
                    if attempt == 0:
                        logger.info(f"[ROUTER] Rate limit attempt 1 — retrying {cfg.provider} in 3s")
                        time.sleep(3)
                        continue
                    _mark_rate_limit(cfg.model)
                    _LAST_ERROR = (
                        f"{cfg.provider} returned a rate-limit error (HTTP 429).\n"
                        f"{'Trying Gemini 1.5 Flash fallback...' if '2.0' in cfg.model else 'Both models rate-limited. Wait 30-60s.'}\n"
                        f"Detail: {exc_str[:200]}"
                    )
                    logger.warning(f"[ROUTER] RATE LIMIT: {_LAST_ERROR}")
                    break   # try next candidate

                # ── TRANSIENT: 500/503/network ────────────────────
                if _is_transient_err(exc):
                    if attempt == 0:
                        logger.info(f"[ROUTER] Transient error — retrying {cfg.provider} in 5s")
                        time.sleep(5)
                        continue
                    # Second attempt also transient → try fallback, no cooldown
                    _LAST_ERROR = (
                        f"{cfg.provider} returned a transient server error.\n"
                        f"Trying fallback model...\n"
                        f"Detail: {exc_str[:200]}"
                    )
                    logger.warning(f"[ROUTER] TRANSIENT: {_LAST_ERROR}")
                    break   # try next candidate without marking rate-limited

                # ── UNKNOWN: log fully, try next candidate ────────
                _LAST_ERROR = (
                    f"Unexpected error from {cfg.provider}.\n"
                    f"Detail: {exc_detail}"
                )
                logger.error(f"[ROUTER] UNKNOWN ERROR: {_LAST_ERROR}")
                break   # try next candidate

    _LAST_ERROR = (
        f"All Gemini models failed. Last error: {_LAST_ERROR}\n\n"
        "Steps to fix:\n"
        "1. Check GEMINI_API_KEY in Streamlit Cloud → Settings → Secrets\n"
        "2. Verify the key at https://aistudio.google.com/apikey\n"
        "3. Click ⚡ Reset All Cooldowns and retry"
    )
    logger.error(f"[ROUTER] ALL MODELS EXHAUSTED. {_LAST_ERROR}")
    return RATE_LIMIT_SENTINEL


# ─────────────────────────────────────────────────────────────
# BACKWARD-COMPATIBLE SHIM
# ─────────────────────────────────────────────────────────────
def call_ai_compat(
    messages    : list,
    temperature : float = 0.1,
    model       : str   = None,
    task        : str   = "code",
) -> str:
    return call_ai(messages, temperature=temperature, task=task)


# ─────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────
def get_router_status() -> list[dict]:
    rows    = []
    has_key = bool(_get_key("GEMINI_API_KEY"))

    for cfg in ALL_MODELS:
        s   = _st(cfg.model)
        now = time.time()

        if not has_key:
            status = "⚪ No API Key — add GEMINI_API_KEY to Streamlit Secrets"
        elif s.auth_failed:
            status = "🔴 Auth Failed — GEMINI_API_KEY is invalid or expired"
        elif s.model_gone:
            status = "⚫ Model Unavailable — will auto-recover on Reset"
        elif now < s.rate_limited_until:
            remaining = int(s.rate_limited_until - now)
            status = f"🟡 Cooldown {remaining}s (hit #{s.rate_limit_count}) — auto-recovers"
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
    if _LAST_USED_PROVIDER != "None yet":
        return _LAST_USED_PROVIDER
    for cfg in ALL_MODELS:
        if _is_available(cfg) and task in cfg.task_types:
            return cfg.provider
    return "No Gemini model available"


def get_active_model(task: str = "code") -> str:
    if _LAST_USED_MODEL != "—":
        return _LAST_USED_MODEL
    for cfg in ALL_MODELS:
        if _is_available(cfg) and task in cfg.task_types:
            return cfg.model.split("/")[-1]
    return "—"


def get_next_provider(task: str = "code") -> str:
    for cfg in ALL_MODELS:
        if _is_available(cfg) and task in cfg.task_types:
            return cfg.provider
    return "None available — check GEMINI_API_KEY"
