"""
ai_router.py  ·  Gemini-Direct Router  ·  v4.1  (NO LITELLM — REST API ONLY)
=============================================================================
FIXES IN v4.1 vs v4.0:
  ✅ FIXED: gemini-1.5-flash → gemini-2.0-flash-lite (1.5-flash removed from v1beta API)
  ✅ FIXED: "gone" branch now explicitly updates _LAST_ERROR (was silently keeping the
             previous model's 429 message → UI falsely showed "rate-limited" for 404 errors)
  ✅ FIXED: show_ai_error() in app.py gets a clean, accurate error string for all branches
  ✅ FIXED: Added "model_not_found" error class for clear 404 UX
  ✅ IMPROVED: model_gone auto-resets after 10 min (transient API changes shouldn't be permanent)
  ✅ RETAINED: All v4.0 direct REST API logic, no litellm needed

ROOT CAUSES FIXED:
  BUG 1 (PRIMARY)  — gemini-1.5-flash was removed from v1beta generateContent endpoint.
                      Every call returned HTTP 404. Fixed by switching to gemini-2.0-flash-lite.
  BUG 2 (SECONDARY) — When model 1 returns 429 and model 2 returns 404, the "gone" branch
                       did NOT update _LAST_ERROR, so the 429 string persisted.
                       The final wrap "All Gemini models failed. Last error: <429 text>" still
                       contained "429" / "rate", triggering the wrong show_ai_error() branch.
                       Fixed by explicitly setting _LAST_ERROR in every error branch.

PROVIDER PRIORITY (v4.1):
  Tier 1 — gemini-2.0-flash       (PRIMARY,  FREE, 1M TPM, 1500 RPD)
  Tier 2 — gemini-2.0-flash-lite  (FALLBACK, FREE, 1.5M TPM, different quota pool)

USAGE (drop-in replacement for v4.0 / v3.x):
  from ai_router import (
      call_ai_compat as call_ai,
      get_router_status, get_active_provider, get_active_model,
      get_next_provider, reset_cooldowns, _get_key, RATE_LIMIT_SENTINEL,
      get_last_error,
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

# Gemini REST base URL  (v1beta supports all current Gemini 2.x / 1.5 pinned models)
_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

# How long a "model_gone" mark persists before auto-retry (seconds).
# 404s can be transient API hiccups; 10 min is conservative but not permanent.
_MODEL_GONE_TTL = 600


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
    model_id    : str
    display     : str
    env_key     : str
    tpm         : int
    rpd         : int
    quality     : int
    cost_per_1k : float
    provider    : str
    task_types  : list = field(default_factory=lambda: ["code", "jira", "summary"])


ALL_MODELS: list[ModelConfig] = [
    ModelConfig(
        # PRIMARY — highest quota, fastest, best quality in free tier
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
        # FALLBACK — different model family = truly independent quota pool
        # gemini-1.5-flash was removed from v1beta in early 2025; use 2.0-flash-lite instead.
        model_id    = "gemini-2.0-flash-lite",
        display     = "gemini/gemini-2.0-flash-lite",
        env_key     = "GEMINI_API_KEY",
        tpm         = 1_500_000,
        rpd         = 1500,
        quality     = 4,
        cost_per_1k = 0.0,
        provider    = "Google Gemini 2.0 Flash Lite",
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
    model_gone_since   : float = 0.0   # ← NEW: for TTL auto-recovery
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
    """Reset all cooldowns and error flags. Returns count of providers recovered."""
    cleared = 0
    for model_id, s in _status_registry.items():
        changed = False
        if s.rate_limited_until > time.time():
            s.rate_limited_until = 0.0
            s.rate_limit_count   = 0
            changed = True
        if s.model_gone:
            s.model_gone       = False
            s.model_gone_since = 0.0
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
    if s.auth_failed:
        return False
    # Auto-recover from model_gone after TTL (transient API 404s)
    if s.model_gone:
        if time.time() - s.model_gone_since > _MODEL_GONE_TTL:
            logger.info(f"[ROUTER] {cfg.model_id} model_gone TTL expired — auto-recovering")
            s.model_gone       = False
            s.model_gone_since = 0.0
        else:
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


def _mark_model_gone(model_id: str):
    s = _st(model_id)
    s.model_gone       = True
    s.model_gone_since = time.time()
    logger.warning(f"[ROUTER] {model_id} marked model_gone (TTL={_MODEL_GONE_TTL}s)")


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
    System message is merged into the first user turn (Gemini v1beta has no system role).
    """
    contents    = []
    system_text = ""

    for msg in messages:
        role    = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            system_text = content
            continue

        gemini_role = "user" if role == "user" else "model"
        contents.append({
            "role"  : gemini_role,
            "parts" : [{"text": content}],
        })

    # Inject system prompt into first user message
    if system_text and contents:
        first = contents[0]
        if first["role"] == "user":
            first["parts"][0]["text"] = system_text + "\n\n" + first["parts"][0]["text"]
    elif system_text:
        contents.insert(0, {"role": "user", "parts": [{"text": system_text}]})

    return {
        "contents": contents,
        "generationConfig": {
            "temperature"    : temperature,
            "maxOutputTokens": max_tokens,
            "topP"           : 0.95,
        },
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
        ],
    }


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
    Error strings always start with HTTP_<code>: for reliable classifier matching.
    """
    import requests as req

    url     = f"{_GEMINI_BASE}/{model_id}:generateContent?key={api_key}"
    payload = _to_gemini_payload(messages, temperature, max_tokens)
    headers = {"Content-Type": "application/json"}

    try:
        resp = req.post(url, json=payload, headers=headers, timeout=60)
    except req.exceptions.ConnectionError as e:
        return False, f"HTTP_CONNECTION_ERROR: {str(e)[:200]}", 0
    except req.exceptions.Timeout:
        return False, "HTTP_TIMEOUT: Request timed out after 60s", 0
    except Exception as e:
        return False, f"HTTP_REQUEST_EXCEPTION: {type(e).__name__}: {str(e)[:200]}", 0

    status = resp.status_code

    if status == 200:
        try:
            data      = resp.json()
            candidate = data["candidates"][0]
            # Check for safety block inside a 200 response
            finish_reason = candidate.get("finishReason", "")
            if finish_reason == "SAFETY":
                return False, "HTTP_200_SAFETY_BLOCK: Content blocked by Gemini safety filters", 0
            text   = candidate["content"]["parts"][0]["text"]
            usage  = data.get("usageMetadata", {})
            tokens = usage.get("totalTokenCount", 0)
            return True, text, tokens
        except (KeyError, IndexError) as e:
            return False, f"HTTP_200_PARSE_ERROR: Could not parse Gemini response — {e}", 0

    # ── Error responses ──────────────────────────────────────
    try:
        err_body = resp.json()
        err_msg  = err_body.get("error", {}).get("message", resp.text[:400])
    except Exception:
        err_msg = resp.text[:400]

    # Always prefix with HTTP_<status>: so _classify_error works reliably
    return False, f"HTTP_{status}: {err_msg}", 0


# ─────────────────────────────────────────────────────────────
# ERROR CLASSIFIERS
# ─────────────────────────────────────────────────────────────
def _classify_error(err: str) -> str:
    """
    Classify a REST error string into a routing decision.
    Returns one of: 'rate_limit' | 'auth' | 'gone' | 'transient' | 'bad_request' | 'safety' | 'unknown'

    IMPORTANT: err strings always start with HTTP_<code>: (from _call_gemini_rest).
    Match on the HTTP code prefix FIRST to avoid substring false-positives.
    e.g. "HTTP_404: ... is not found" must NOT match "rate" or "429".
    """
    e = err.lower()

    # ── Auth / permission (401, 403) ──────────────────────────
    if any(x in e for x in [
        "http_401", "http_403",
        "invalid api key", "api key not valid", "api_key_invalid",
        "key not valid", "authentication", "unauthorized", "permission_denied",
    ]):
        return "auth"

    # ── Rate limit / quota (429) ──────────────────────────────
    if any(x in e for x in [
        "http_429",
        "resource_exhausted", "too many requests",
    ]):
        return "rate_limit"
    # Check "rate limit" and "quota" only when NOT a 404 (avoid false-matches on model names)
    if "http_404" not in e and "http_4" not in e[:10]:
        if "rate limit" in e or "quota" in e:
            return "rate_limit"

    # ── Model not found / deprecated (404, unsupported) ──────
    if any(x in e for x in [
        "http_404",
        "is not found for api version",
        "not supported for generatecontent",
        "decommissioned", "no longer supported", "unsupported model",
        "does not exist",
    ]):
        return "gone"

    # ── Bad request (400) ─────────────────────────────────────
    if any(x in e for x in [
        "http_400", "bad request", "badrequest",
        "invalid_request", "invalid argument",
        "http_200_parse_error",
    ]):
        return "bad_request"

    # ── Transient server errors (5xx, network) ────────────────
    if any(x in e for x in [
        "http_500", "http_502", "http_503", "http_504",
        "http_connection_error", "http_timeout",
        "service unavailable", "internal server error",
    ]):
        return "transient"

    # ── Safety block ──────────────────────────────────────────
    if "safety_block" in e or "safety filter" in e:
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
    Call best available Gemini model via REST API.
    Returns response text, or RATE_LIMIT_SENTINEL if all models are exhausted.
    _LAST_ERROR is ALWAYS updated with a precise, UI-friendly message on any failure.
    """
    global _LAST_ERROR

    api_key     = _get_key("GEMINI_API_KEY")
    token_limit = TASK_TOKENS.get(task, max_tokens)

    if not api_key:
        _LAST_ERROR = (
            "NO_API_KEY: GEMINI_API_KEY is not set.\n"
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
        # Don't wrap a previous error — set a fresh, clear message
        _LAST_ERROR = (
            "ALL_UNAVAILABLE: All Gemini models are in cooldown or unavailable.\n"
            "Click ⚡ Reset All Cooldowns and try again, or wait for the cooldown to expire."
        )
        logger.error(f"[ROUTER] No candidates. {_LAST_ERROR}")
        return RATE_LIMIT_SENTINEL

    logger.info(f"[ROUTER] Candidates for task={task}: {[c.provider for c in candidates]}")

    last_error_for_exhaustion = ""  # Track the LAST model's error for final message

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

            # ── Failed — classify and route ──────────────────
            error_class = _classify_error(content)
            logger.warning(
                f"[ROUTER] {cfg.provider} [{error_class}] attempt={attempt}: {content[:200]}"
            )

            # ── Auth failure ──────────────────────────────────
            if error_class == "auth":
                _LAST_ERROR = (
                    f"AUTH_ERROR: Authentication failed for {cfg.provider}.\n"
                    f"Your GEMINI_API_KEY is invalid or expired.\n"
                    f"→ Go to Streamlit Cloud → Settings → Secrets and update GEMINI_API_KEY\n"
                    f"→ Get a new free key at https://aistudio.google.com/apikey\n"
                    f"Detail: {content[:300]}"
                )
                _st(cfg.model_id).auth_failed = True
                last_error_for_exhaustion = _LAST_ERROR
                logger.error(f"[ROUTER] AUTH FAIL: {_LAST_ERROR}")
                break  # try next model

            # ── Model not found / deprecated ──────────────────
            elif error_class == "gone":
                _LAST_ERROR = (
                    f"MODEL_NOT_FOUND: {cfg.provider} — model '{cfg.model_id}' returned HTTP 404.\n"
                    f"This model may have been deprecated by Google.\n"
                    f"Trying next available model automatically...\n"
                    f"Detail: {content[:300]}"
                )
                _mark_model_gone(cfg.model_id)
                last_error_for_exhaustion = _LAST_ERROR
                logger.warning(f"[ROUTER] MODEL GONE: {_LAST_ERROR}")
                break  # try next model

            # ── Bad request ───────────────────────────────────
            elif error_class == "bad_request":
                _LAST_ERROR = (
                    f"BAD_REQUEST: {cfg.provider} rejected the request (HTTP 400).\n"
                    f"This may be a temporary API issue. Trying fallback model...\n"
                    f"Detail: {content[:300]}"
                )
                _mark_model_gone(cfg.model_id)
                last_error_for_exhaustion = _LAST_ERROR
                logger.warning(f"[ROUTER] BAD REQUEST: {_LAST_ERROR}")
                break  # try next model

            # ── Rate limit ────────────────────────────────────
            elif error_class == "rate_limit":
                if attempt < max_attempts:
                    logger.info(f"[ROUTER] Rate limit — retrying {cfg.provider} in 3s...")
                    time.sleep(3)
                    continue
                _mark_rate_limit(cfg.model_id)
                _LAST_ERROR = (
                    f"RATE_LIMIT: {cfg.provider} is rate-limited (HTTP 429).\n"
                    f"{'Trying Gemini 2.0 Flash Lite fallback...' if 'lite' not in cfg.model_id.lower() else 'Both models rate-limited. Wait 30–60s then click ⚡ Reset All Cooldowns.'}\n"
                    f"Detail: {content[:200]}"
                )
                last_error_for_exhaustion = _LAST_ERROR
                logger.warning(f"[ROUTER] RATE LIMIT: {_LAST_ERROR}")
                break  # try next model

            # ── Transient server error ────────────────────────
            elif error_class == "transient":
                if attempt < max_attempts:
                    logger.info(f"[ROUTER] Transient error — retrying {cfg.provider} in 5s...")
                    time.sleep(5)
                    continue
                _LAST_ERROR = (
                    f"SERVER_ERROR: {cfg.provider} returned a temporary server error.\n"
                    f"Trying fallback model...\n"
                    f"Detail: {content[:200]}"
                )
                last_error_for_exhaustion = _LAST_ERROR
                logger.warning(f"[ROUTER] TRANSIENT (exhausted): {_LAST_ERROR}")
                break  # try next model (no cooldown — it's transient)

            # ── Safety block ──────────────────────────────────
            elif error_class == "safety":
                _LAST_ERROR = (
                    f"SAFETY_BLOCK: {cfg.provider} blocked the response for safety reasons.\n"
                    f"Try rephrasing your prompt."
                )
                logger.warning(f"[ROUTER] SAFETY BLOCK: {_LAST_ERROR}")
                return RATE_LIMIT_SENTINEL  # Same content will be blocked on other models

            # ── Unknown ───────────────────────────────────────
            else:
                _LAST_ERROR = (
                    f"UNKNOWN_ERROR: Unexpected error from {cfg.provider}.\n"
                    f"Detail: {content[:300]}"
                )
                last_error_for_exhaustion = _LAST_ERROR
                logger.error(f"[ROUTER] UNKNOWN: {_LAST_ERROR}")
                if attempt < max_attempts:
                    time.sleep(2)
                    continue
                break  # try next model

    # ── All candidates exhausted ──────────────────────────────
    # Use the last real error (NOT a wrapped version that re-embeds old error strings)
    _LAST_ERROR = last_error_for_exhaustion or _LAST_ERROR
    logger.error(f"[ROUTER] ALL MODELS EXHAUSTED. Last error: {_LAST_ERROR}")
    return RATE_LIMIT_SENTINEL


# ─────────────────────────────────────────────────────────────
# BACKWARD-COMPATIBLE SHIM (drop-in for v4.0 / v3.x)
# ─────────────────────────────────────────────────────────────
def call_ai_compat(
    messages    : list,
    temperature : float = 0.1,
    model       : str   = None,
    task        : str   = "code",
) -> str:
    return call_ai(messages, temperature=temperature, task=task)


# ─────────────────────────────────────────────────────────────
# UI HELPERS (same interface as v4.0 / v3.x)
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
            remaining_ttl = int(_MODEL_GONE_TTL - (now - s.model_gone_since))
            status = f"⚫ Model 404 — auto-recovers in {max(remaining_ttl, 0)}s or click Reset"
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
