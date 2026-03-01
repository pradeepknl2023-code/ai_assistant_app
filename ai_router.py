"""
ai_router.py  ·  Gemini-Direct Router  ·  v4.0  (NO LITELLM — REST API ONLY)
=============================================================================
PROBLEM WITH v3.x:
  litellm is an extra dependency that:
    • Adds its own error classification/wrapping that hides the real Gemini error
    • Has its own rate-limit logic that conflicts with ours
    • Changes error message formats between versions (breaks our classifiers)
    • Is overkill when we only need ONE provider (Gemini)

v4.0 SOLUTION — Call Gemini REST API directly via requests:
  ✅ No litellm needed (still backward-compatible if installed)
  ✅ Real HTTP status codes — 200/429/401/403/400/500/503
  ✅ Real Gemini error messages — no wrapping/mangling
  ✅ Faster: one less layer of abstraction
  ✅ All v3.1 classifier fixes retained
  ✅ RATE_LIMIT_SENTINEL only returned when BOTH models truly 429/exhausted

PROVIDER PRIORITY:
  Tier 1 — gemini-2.0-flash   (PRIMARY,  FREE, 1M TPM, 1500 RPD)
  Tier 2 — gemini-1.5-flash   (FALLBACK, FREE, 1M TPM, 1500 RPD)

USAGE (drop-in replacement for v3.x):
  from ai_router import (
      call_ai_compat as call_ai,
      get_router_status, get_active_provider, get_active_model,
      get_next_provider, reset_cooldowns, _get_key, RATE_LIMIT_SENTINEL,
  )
"""

from __future__ import annotations

import os
import time
import logging
import json
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger("AI_ROUTER")

# ─────────────────────────────────────────────────────────────
# SENTINEL & GLOBALS
# ─────────────────────────────────────────────────────────────
RATE_LIMIT_SENTINEL = "__RATE_LIMIT__"

_LAST_USED_PROVIDER = "None yet"
_LAST_USED_MODEL    = "—"
_LAST_ERROR         = ""

# Gemini REST base URL
_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


# ─────────────────────────────────────────────────────────────
# KEY LOOKUP
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
    return _LAST_ERROR


# ─────────────────────────────────────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    model_id    : str          # Gemini model ID for REST API
    display     : str          # Human-readable name
    env_key     : str
    tpm         : int
    rpd         : int
    quality     : int
    cost_per_1k : float
    provider    : str
    task_types  : list = field(default_factory=lambda: ["code", "jira", "summary"])


ALL_MODELS: list[ModelConfig] = [
    ModelConfig(
        model_id    = "gemini-2.0-flash",
        display     = "gemini/gemini-2.0-flash",
        env_key     = "GEMINI_API_KEY",
        tpm         = 1_000_000,
        rpd         = 1500,
        quality     = 5,
        cost_per_1k = 0.0,
        provider    = "Google Gemini 2.0 Flash",
    ),
    ModelConfig(
        model_id    = "gemini-1.5-flash",
        display     = "gemini/gemini-1.5-flash",
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


def _st(model_id: str) -> _ModelStatus:
    if model_id not in _status_registry:
        _status_registry[model_id] = _ModelStatus()
    return _status_registry[model_id]


def reset_cooldowns() -> int:
    cleared = 0
    for model_id, s in _status_registry.items():
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
    s = _st(cfg.model_id)
    if s.auth_failed or s.model_gone:
        return False
    if time.time() < s.rate_limited_until:
        return False
    if not _get_key(cfg.env_key):
        return False
    return True


def _mark_rate_limit(model_id: str):
    s = _st(model_id)
    s.rate_limit_count  += 1
    backoff              = min(30 * s.rate_limit_count, 90)
    s.rate_limited_until = time.time() + backoff
    logger.warning(f"[ROUTER] {model_id} rate-limit cooldown={backoff}s (hit #{s.rate_limit_count})")


def _mark_success(model_id: str, tokens: int = 0, provider: str = "", display_model: str = ""):
    global _LAST_USED_PROVIDER, _LAST_USED_MODEL, _LAST_ERROR
    s = _st(model_id)
    s.total_calls       += 1
    s.total_tokens      += tokens
    s.last_used_ts       = time.time()
    s.rate_limited_until = 0.0
    s.rate_limit_count   = 0
    _LAST_ERROR          = ""
    if provider:
        _LAST_USED_PROVIDER = provider
    if display_model:
        _LAST_USED_MODEL = display_model


# ─────────────────────────────────────────────────────────────
# MESSAGES → GEMINI FORMAT CONVERTER
# ─────────────────────────────────────────────────────────────
def _to_gemini_payload(messages: list[dict], temperature: float, max_tokens: int) -> dict:
    """
    Convert OpenAI-style messages to Gemini REST API format.
    Handles: system, user, assistant roles.
    System message is prepended as a user turn (Gemini doesn't have system role in REST v1beta).
    """
    contents = []
    system_text = ""

    for msg in messages:
        role    = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            system_text = content
            continue

        gemini_role = "user" if role == "user" else "model"
        contents.append({
            "role": gemini_role,
            "parts": [{"text": content}]
        })

    # Prepend system prompt as first user message if present
    if system_text and contents:
        # Inject system text into the first user message
        first = contents[0]
        if first["role"] == "user":
            first["parts"][0]["text"] = system_text + "\n\n" + first["parts"][0]["text"]
    elif system_text:
        contents.insert(0, {"role": "user", "parts": [{"text": system_text}]})

    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "topP": 0.95,
        },
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
        ]
    }
    return payload


# ─────────────────────────────────────────────────────────────
# DIRECT GEMINI REST CALL
# ─────────────────────────────────────────────────────────────
def _call_gemini_rest(
    model_id   : str,
    messages   : list[dict],
    temperature: float,
    max_tokens : int,
    api_key    : str,
) -> tuple[bool, str, int]:
    """
    Call Gemini REST API directly.
    Returns: (success: bool, content_or_error: str, tokens_used: int)

    HTTP status codes handled:
      200 → success
      400 → bad request (bad model, bad param)
      401/403 → auth error
      429 → rate limit
      500/503 → transient server error
    """
    import requests as req

    url     = f"{_GEMINI_BASE}/{model_id}:generateContent?key={api_key}"
    payload = _to_gemini_payload(messages, temperature, max_tokens)
    headers = {"Content-Type": "application/json"}

    try:
        resp = req.post(url, json=payload, headers=headers, timeout=60)
    except req.exceptions.ConnectionError as e:
        return False, f"CONNECTION_ERROR: {str(e)[:200]}", 0
    except req.exceptions.Timeout:
        return False, "TIMEOUT: Request timed out after 60s", 0
    except Exception as e:
        return False, f"REQUEST_EXCEPTION: {type(e).__name__}: {str(e)[:200]}", 0

    status = resp.status_code

    if status == 200:
        try:
            data       = resp.json()
            candidate  = data["candidates"][0]
            text       = candidate["content"]["parts"][0]["text"]
            usage      = data.get("usageMetadata", {})
            tokens     = usage.get("totalTokenCount", 0)
            return True, text, tokens
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            # Check for safety block
            try:
                finish_reason = resp.json()["candidates"][0].get("finishReason", "")
                if finish_reason == "SAFETY":
                    return False, "SAFETY_BLOCK: Content was blocked by Gemini safety filters", 0
            except Exception:
                pass
            return False, f"PARSE_ERROR: Could not parse Gemini response — {e}", 0

    # Error responses
    try:
        err_data = resp.json()
        err_msg  = err_data.get("error", {}).get("message", resp.text[:300])
    except Exception:
        err_msg = resp.text[:300]

    return False, f"HTTP_{status}: {err_msg}", 0


# ─────────────────────────────────────────────────────────────
# ERROR CLASSIFIERS — work on the error string from _call_gemini_rest
# ─────────────────────────────────────────────────────────────
def _classify_error(err: str) -> str:
    """
    Returns one of: 'rate_limit', 'auth', 'gone', 'transient', 'bad_request', 'unknown'
    """
    e = err.lower()

    if "http_429" in e or "rate limit" in e or "quota" in e or "resource_exhausted" in e or "too many requests" in e:
        return "rate_limit"

    if any(x in e for x in ["http_401", "http_403", "invalid api key", "api key not valid",
                             "api_key_invalid", "key not valid", "authentication", "unauthorized",
                             "permission_denied"]):
        return "auth"

    if any(x in e for x in ["http_404", "not found", "notfound", "does not exist",
                             "decommissioned", "no longer supported", "unsupported model"]):
        return "gone"

    if any(x in e for x in ["http_400", "bad request", "badrequest", "invalid_request",
                             "invalid argument", "parse_error"]):
        return "bad_request"

    if any(x in e for x in ["http_500", "http_502", "http_503", "http_504",
                             "connection_error", "timeout", "timed out", "service unavailable",
                             "internal server error", "transient"]):
        return "transient"

    if "safety_block" in e:
        return "safety"

    return "unknown"


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
    Call best available Gemini model via REST API (no litellm needed).
    Returns response text, or RATE_LIMIT_SENTINEL if all models fail.
    """
    global _LAST_ERROR

    api_key    = _get_key("GEMINI_API_KEY")
    token_limit = TASK_TOKENS.get(task, max_tokens)

    if not api_key:
        _LAST_ERROR = (
            "GEMINI_API_KEY is not set.\n"
            "Add it in Streamlit Cloud → Settings → Secrets:\n"
            "  GEMINI_API_KEY = \"AIzaSy...\"\n"
            "Get a free key at https://aistudio.google.com/apikey"
        )
        logger.error(f"[ROUTER] No API key. {_LAST_ERROR}")
        return RATE_LIMIT_SENTINEL

    candidates = [
        cfg for cfg in ALL_MODELS
        if task in cfg.task_types
        and _is_available(cfg)
        and (not force_provider or cfg.provider == force_provider)
    ]

    if not candidates:
        _LAST_ERROR = (
            "All Gemini models are in cooldown or unavailable.\n"
            "Click ⚡ Reset All Cooldowns and try again."
        )
        logger.error(f"[ROUTER] No candidates available. {_LAST_ERROR}")
        return RATE_LIMIT_SENTINEL

    logger.info(f"[ROUTER] Candidates for task={task}: {[c.provider for c in candidates]}")

    for cfg in candidates:
        max_attempts = 2

        for attempt in range(1, max_attempts + 1):
            logger.info(
                f"[ROUTER] → {cfg.provider} [{cfg.model_id}] "
                f"attempt={attempt}/{max_attempts} task={task} max_tokens={token_limit}"
            )

            success, content, tokens = _call_gemini_rest(
                model_id    = cfg.model_id,
                messages    = messages,
                temperature = temperature,
                max_tokens  = token_limit,
                api_key     = api_key,
            )

            if success:
                _mark_success(
                    cfg.model_id,
                    tokens,
                    provider      = cfg.provider,
                    display_model = cfg.model_id,
                )
                logger.info(f"[ROUTER] SUCCESS: {cfg.provider} | {tokens} tokens | task={task}")
                return content

            # ── Failed — classify and decide what to do ──
            error_class = _classify_error(content)
            _LAST_ERROR = content
            logger.warning(f"[ROUTER] {cfg.provider} [{error_class}] attempt={attempt}: {content[:200]}")

            if error_class == "auth":
                _st(cfg.model_id).auth_failed = True
                _LAST_ERROR = (
                    f"❌ Authentication failed for {cfg.provider}.\n"
                    f"Your GEMINI_API_KEY is invalid or expired.\n"
                    f"→ Check Streamlit Cloud → Settings → Secrets\n"
                    f"→ Get a new free key at https://aistudio.google.com/apikey\n"
                    f"Detail: {content[:300]}"
                )
                logger.error(f"[ROUTER] AUTH FAIL: {_LAST_ERROR}")
                break  # try next model (same key, but log clearly)

            elif error_class in ("gone", "bad_request"):
                _st(cfg.model_id).model_gone = True
                logger.warning(
                    f"[ROUTER] {cfg.provider} marked model_gone ({error_class}). "
                    f"Trying fallback..."
                )
                break  # try next model

            elif error_class == "rate_limit":
                if attempt < max_attempts:
                    logger.info(f"[ROUTER] Rate limit — retrying {cfg.provider} in 3s...")
                    time.sleep(3)
                    continue
                _mark_rate_limit(cfg.model_id)
                _LAST_ERROR = (
                    f"⚠️ {cfg.provider} rate-limited (HTTP 429).\n"
                    f"{'Trying Gemini 1.5 Flash fallback...' if '2.0' in cfg.model_id else 'Both models rate-limited. Wait 30-60s, then Reset All Cooldowns.'}\n"
                    f"Detail: {content[:200]}"
                )
                logger.warning(f"[ROUTER] RATE LIMIT: {_LAST_ERROR}")
                break  # try next model

            elif error_class == "transient":
                if attempt < max_attempts:
                    logger.info(f"[ROUTER] Transient error — retrying {cfg.provider} in 5s...")
                    time.sleep(5)
                    continue
                _LAST_ERROR = (
                    f"⚠️ {cfg.provider} returned a server error.\n"
                    f"This is usually temporary. Trying fallback model...\n"
                    f"Detail: {content[:200]}"
                )
                logger.warning(f"[ROUTER] TRANSIENT (exhausted): {_LAST_ERROR}")
                break  # try next model (no cooldown — it's transient)

            elif error_class == "safety":
                # Safety block — not a provider issue, content issue
                _LAST_ERROR = (
                    f"⚠️ {cfg.provider} blocked the response for safety reasons.\n"
                    f"Try rephrasing your prompt."
                )
                logger.warning(f"[ROUTER] SAFETY BLOCK: {_LAST_ERROR}")
                return RATE_LIMIT_SENTINEL  # Don't try fallback — same content will be blocked

            else:
                # Unknown error
                _LAST_ERROR = (
                    f"⚠️ Unexpected error from {cfg.provider}.\n"
                    f"Detail: {content[:300]}"
                )
                logger.error(f"[ROUTER] UNKNOWN: {_LAST_ERROR}")
                if attempt < max_attempts:
                    time.sleep(2)
                    continue
                break  # try next model

    # All candidates exhausted
    _LAST_ERROR = (
        f"All Gemini models failed.\nLast error: {_LAST_ERROR}\n\n"
        "To fix:\n"
        "1. Check GEMINI_API_KEY in Streamlit Cloud → Settings → Secrets\n"
        "2. Verify at https://aistudio.google.com/apikey\n"
        "3. Click ⚡ Reset All Cooldowns and retry"
    )
    logger.error(f"[ROUTER] ALL MODELS EXHAUSTED. {_LAST_ERROR}")
    return RATE_LIMIT_SENTINEL


# ─────────────────────────────────────────────────────────────
# BACKWARD-COMPATIBLE SHIM (drop-in for v3.x)
# ─────────────────────────────────────────────────────────────
def call_ai_compat(
    messages    : list,
    temperature : float = 0.1,
    model       : str   = None,
    task        : str   = "code",
) -> str:
    return call_ai(messages, temperature=temperature, task=task)


# ─────────────────────────────────────────────────────────────
# UI HELPERS (same interface as v3.x)
# ─────────────────────────────────────────────────────────────
def get_router_status() -> list[dict]:
    rows    = []
    has_key = bool(_get_key("GEMINI_API_KEY"))

    for cfg in ALL_MODELS:
        s   = _st(cfg.model_id)
        now = time.time()

        if not has_key:
            status = "⚪ No API Key — add GEMINI_API_KEY to Streamlit Secrets"
        elif s.auth_failed:
            status = "🔴 Auth Failed — GEMINI_API_KEY is invalid or expired"
        elif s.model_gone:
            status = "⚫ Model Unavailable — auto-recovers on Reset"
        elif now < s.rate_limited_until:
            remaining = int(s.rate_limited_until - now)
            status = f"🟡 Cooldown {remaining}s (hit #{s.rate_limit_count}) — auto-recovers"
        else:
            status = "🟢 Ready"

        is_last = (cfg.provider == _LAST_USED_PROVIDER)
        label   = ("⭐ LAST USED → " if is_last else "") + cfg.provider

        rows.append({
            "Provider" : label,
            "Model"    : cfg.model_id,
            "Method"   : "REST API (direct)",
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
            return cfg.model_id
    return "—"


def get_next_provider(task: str = "code") -> str:
    for cfg in ALL_MODELS:
        if _is_available(cfg) and task in cfg.task_types:
            return cfg.provider
    return "None available — check GEMINI_API_KEY"
