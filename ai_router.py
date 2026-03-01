"""
ai_router.py  ·  Gemini-Direct Router  ·  v4.2  (5-MODEL POOL — NEAR-ZERO DOWNTIME)
=====================================================================================
PROBLEM FIXED IN v4.2:
  With only 2 models, bursts of usage exhausted both simultaneously → app dead.

v4.2 SOLUTION — 5 Free Gemini Models, Each With Independent Quota:
  Tier 1 — gemini-2.5-flash-preview-05-20  (best quality, free preview)
  Tier 2 — gemini-2.0-flash                (proven workhorse, 1500 RPD)
  Tier 3 — gemini-2.0-flash-lite           (fast, 1.5M TPM)
  Tier 4 — gemini-1.5-flash-8b             (tiny, independent quota bucket)
  Tier 5 — gemini-1.5-flash-001            (pinned stable, unlike unpinned which 404s)

  Each model has its OWN rate-limit counter at Google.
  Probability of ALL 5 being rate-limited simultaneously: near zero.

RETAINED FROM v4.1:
  ✅ Direct REST API (no litellm needed)
  ✅ Structured error prefix tokens — RATE_LIMIT:, AUTH_ERROR:, MODEL_NOT_FOUND:
  ✅ _LAST_ERROR explicitly set in every error branch (no silent masking)
  ✅ model_gone TTL auto-recovery (10 min)
"""

from __future__ import annotations

import os
import time
import logging
import json
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger("AI_ROUTER")

RATE_LIMIT_SENTINEL = "__RATE_LIMIT__"
_LAST_USED_PROVIDER = "None yet"
_LAST_USED_MODEL    = "—"
_LAST_ERROR         = ""
_GEMINI_BASE        = "https://generativelanguage.googleapis.com/v1beta/models"
_MODEL_GONE_TTL     = 600


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
        return val
    except Exception:
        return ""


def get_last_error() -> str:
    return _LAST_ERROR


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
        model_id="gemini-2.5-flash-preview-05-20", display="gemini/gemini-2.5-flash-preview",
        env_key="GEMINI_API_KEY", tpm=1_000_000, rpd=500, quality=5, cost_per_1k=0.0,
        provider="Google Gemini 2.5 Flash Preview",
    ),
    ModelConfig(
        model_id="gemini-2.0-flash", display="gemini/gemini-2.0-flash",
        env_key="GEMINI_API_KEY", tpm=1_000_000, rpd=1500, quality=5, cost_per_1k=0.0,
        provider="Google Gemini 2.0 Flash",
    ),
    ModelConfig(
        model_id="gemini-2.0-flash-lite", display="gemini/gemini-2.0-flash-lite",
        env_key="GEMINI_API_KEY", tpm=1_500_000, rpd=1500, quality=4, cost_per_1k=0.0,
        provider="Google Gemini 2.0 Flash Lite",
    ),
    ModelConfig(
        model_id="gemini-1.5-flash-8b", display="gemini/gemini-1.5-flash-8b",
        env_key="GEMINI_API_KEY", tpm=1_000_000, rpd=1500, quality=3, cost_per_1k=0.0,
        provider="Google Gemini 1.5 Flash 8B",
    ),
    ModelConfig(
        model_id="gemini-1.5-flash-001", display="gemini/gemini-1.5-flash-001",
        env_key="GEMINI_API_KEY", tpm=1_000_000, rpd=1500, quality=3, cost_per_1k=0.0,
        provider="Google Gemini 1.5 Flash 001",
    ),
]

TASK_TOKENS = {"code": 1200, "summary": 350, "jira": 4000}


@dataclass
class _ModelStatus:
    rate_limited_until : float = 0.0
    auth_failed        : bool  = False
    model_gone         : bool  = False
    model_gone_since   : float = 0.0
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
            s.model_gone_since = 0.0
            changed = True
        if s.auth_failed:
            s.auth_failed = False
            changed = True
        if changed:
            cleared += 1
    logger.info(f"[ROUTER] reset_cooldowns — {cleared} recovered")
    return cleared


def _is_available(cfg: ModelConfig) -> bool:
    s = _st(cfg.model_id)
    if s.auth_failed:
        return False
    if s.model_gone:
        if time.time() - s.model_gone_since > _MODEL_GONE_TTL:
            s.model_gone = False
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
    s.rate_limit_count += 1
    backoff = min(30 * s.rate_limit_count, 90)
    s.rate_limited_until = time.time() + backoff
    logger.warning(f"[ROUTER] {model_id} rate-limited cooldown={backoff}s (#{s.rate_limit_count})")


def _mark_model_gone(model_id: str):
    s = _st(model_id)
    s.model_gone = True
    s.model_gone_since = time.time()


def _mark_success(model_id: str, tokens: int = 0, provider: str = "", display_model: str = ""):
    global _LAST_USED_PROVIDER, _LAST_USED_MODEL, _LAST_ERROR
    s = _st(model_id)
    s.total_calls += 1
    s.total_tokens += tokens
    s.last_used_ts = time.time()
    s.rate_limited_until = 0.0
    s.rate_limit_count = 0
    _LAST_ERROR = ""
    if provider:
        _LAST_USED_PROVIDER = provider
    if display_model:
        _LAST_USED_MODEL = display_model


def _to_gemini_payload(messages: list[dict], temperature: float, max_tokens: int) -> dict:
    contents = []
    system_text = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            system_text = content
            continue
        gemini_role = "user" if role == "user" else "model"
        contents.append({"role": gemini_role, "parts": [{"text": content}]})
    if system_text and contents:
        first = contents[0]
        if first["role"] == "user":
            first["parts"][0]["text"] = system_text + "\n\n" + first["parts"][0]["text"]
    elif system_text:
        contents.insert(0, {"role": "user", "parts": [{"text": system_text}]})
    return {
        "contents": contents,
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens, "topP": 0.95},
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
        ],
    }


def _call_gemini_rest(model_id, messages, temperature, max_tokens, api_key):
    import requests as req
    url = f"{_GEMINI_BASE}/{model_id}:generateContent?key={api_key}"
    payload = _to_gemini_payload(messages, temperature, max_tokens)
    try:
        resp = req.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=60)
    except req.exceptions.ConnectionError as e:
        return False, f"HTTP_CONNECTION_ERROR: {str(e)[:200]}", 0
    except req.exceptions.Timeout:
        return False, "HTTP_TIMEOUT: Request timed out after 60s", 0
    except Exception as e:
        return False, f"HTTP_REQUEST_EXCEPTION: {type(e).__name__}: {str(e)[:200]}", 0

    if resp.status_code == 200:
        try:
            data = resp.json()
            candidate = data["candidates"][0]
            if candidate.get("finishReason") == "SAFETY":
                return False, "HTTP_200_SAFETY_BLOCK: Content blocked by safety filters", 0
            text = candidate["content"]["parts"][0]["text"]
            tokens = data.get("usageMetadata", {}).get("totalTokenCount", 0)
            return True, text, tokens
        except (KeyError, IndexError) as e:
            return False, f"HTTP_200_PARSE_ERROR: {e}", 0

    try:
        err_msg = resp.json().get("error", {}).get("message", resp.text[:400])
    except Exception:
        err_msg = resp.text[:400]
    return False, f"HTTP_{resp.status_code}: {err_msg}", 0


def _classify_error(err: str) -> str:
    e = err.lower()
    if any(x in e for x in ["http_401", "http_403", "invalid api key", "api key not valid",
                              "authentication", "unauthorized", "permission_denied"]):
        return "auth"
    if "http_429" in e or "resource_exhausted" in e or "too many requests" in e:
        return "rate_limit"
    if "http_404" not in e and ("rate limit" in e or "quota" in e):
        return "rate_limit"
    if any(x in e for x in ["http_404", "is not found for api version",
                              "not supported for generatecontent", "decommissioned"]):
        return "gone"
    if any(x in e for x in ["http_400", "bad request", "invalid argument", "http_200_parse_error"]):
        return "bad_request"
    if any(x in e for x in ["http_500", "http_502", "http_503", "http_504",
                              "http_connection_error", "http_timeout", "service unavailable"]):
        return "transient"
    if "safety_block" in e or "safety filter" in e:
        return "safety"
    return "unknown"


def call_ai(
    messages      : list[dict],
    temperature   : float = 0.1,
    max_tokens    : int   = 1200,
    task          : Literal["code", "summary", "jira"] = "code",
    require_free  : bool  = False,
    force_provider: str | None = None,
) -> str:
    global _LAST_ERROR

    api_key = _get_key("GEMINI_API_KEY")
    token_limit = TASK_TOKENS.get(task, max_tokens)

    if not api_key:
        _LAST_ERROR = (
            "NO_API_KEY: GEMINI_API_KEY is not set.\n"
            "Add it in Streamlit Cloud → Settings → Secrets:\n"
            "  GEMINI_API_KEY = \"AIzaSy...\"\n"
            "Get a free key at https://aistudio.google.com/apikey"
        )
        return RATE_LIMIT_SENTINEL

    candidates = [
        cfg for cfg in ALL_MODELS
        if task in cfg.task_types
        and _is_available(cfg)
        and (not force_provider or cfg.provider == force_provider)
    ]

    if not candidates:
        _LAST_ERROR = (
            "ALL_UNAVAILABLE: All 5 Gemini models are in cooldown or unavailable.\n"
            "Click ⚡ Reset All Cooldowns and try again."
        )
        return RATE_LIMIT_SENTINEL

    logger.info(f"[ROUTER] {len(candidates)} candidates: {[c.provider for c in candidates]}")
    last_error_for_exhaustion = ""

    for cfg in candidates:
        for attempt in range(1, 3):
            logger.info(f"[ROUTER] → {cfg.provider} attempt={attempt}/2 task={task}")
            success, content, tokens = _call_gemini_rest(
                cfg.model_id, messages, temperature, token_limit, api_key
            )

            if success:
                _mark_success(cfg.model_id, tokens, provider=cfg.provider, display_model=cfg.model_id)
                logger.info(f"[ROUTER] ✅ {cfg.provider} | {tokens} tokens")
                return content

            error_class = _classify_error(content)
            logger.warning(f"[ROUTER] ❌ {cfg.provider} [{error_class}]: {content[:120]}")

            if error_class == "auth":
                _LAST_ERROR = (
                    f"AUTH_ERROR: GEMINI_API_KEY is invalid or expired.\n"
                    f"Get a new key at https://aistudio.google.com/apikey\n"
                    f"Update: Streamlit Cloud → Settings → Secrets → GEMINI_API_KEY\n"
                    f"Detail: {content[:300]}"
                )
                _st(cfg.model_id).auth_failed = True
                last_error_for_exhaustion = _LAST_ERROR
                break

            elif error_class in ("gone", "bad_request"):
                _LAST_ERROR = (
                    f"MODEL_NOT_FOUND: {cfg.provider} returned HTTP 404/400.\n"
                    f"Trying next of {len(candidates)-1} remaining models...\n"
                    f"Detail: {content[:300]}"
                )
                _mark_model_gone(cfg.model_id)
                last_error_for_exhaustion = _LAST_ERROR
                break

            elif error_class == "rate_limit":
                if attempt < 2:
                    time.sleep(2)
                    continue
                _mark_rate_limit(cfg.model_id)
                remaining = len([c for c in candidates if c.model_id != cfg.model_id and _is_available(c)])
                _LAST_ERROR = (
                    f"RATE_LIMIT: {cfg.provider} is rate-limited (HTTP 429).\n"
                    f"{remaining} model(s) still available — trying them now...\n"
                    f"Detail: {content[:200]}"
                ) if remaining > 0 else (
                    f"RATE_LIMIT: All models rate-limited (HTTP 429).\n"
                    f"Wait 30–60s then click ⚡ Reset All Cooldowns.\n"
                    f"Detail: {content[:200]}"
                )
                last_error_for_exhaustion = _LAST_ERROR
                break

            elif error_class == "transient":
                if attempt < 2:
                    time.sleep(5)
                    continue
                _LAST_ERROR = f"SERVER_ERROR: {cfg.provider} server error. Trying next...\nDetail: {content[:200]}"
                last_error_for_exhaustion = _LAST_ERROR
                break

            elif error_class == "safety":
                _LAST_ERROR = f"SAFETY_BLOCK: Content blocked. Try rephrasing your prompt."
                return RATE_LIMIT_SENTINEL

            else:
                _LAST_ERROR = f"UNKNOWN_ERROR: {cfg.provider}.\nDetail: {content[:300]}"
                last_error_for_exhaustion = _LAST_ERROR
                if attempt < 2:
                    time.sleep(2)
                    continue
                break

    _LAST_ERROR = last_error_for_exhaustion or _LAST_ERROR
    logger.error(f"[ROUTER] ALL MODELS EXHAUSTED.")
    return RATE_LIMIT_SENTINEL


def call_ai_compat(messages, temperature=0.1, model=None, task="code") -> str:
    return call_ai(messages, temperature=temperature, task=task)


def get_router_status() -> list[dict]:
    rows = []
    has_key = bool(_get_key("GEMINI_API_KEY"))
    for cfg in ALL_MODELS:
        s = _st(cfg.model_id)
        now = time.time()
        if not has_key:
            status = "⚪ No API Key"
        elif s.auth_failed:
            status = "🔴 Auth Failed — key invalid/expired"
        elif s.model_gone:
            ttl_left = int(_MODEL_GONE_TTL - (now - s.model_gone_since))
            status = f"⚫ Model 404 — auto-recovers in {max(ttl_left,0)}s"
        elif now < s.rate_limited_until:
            status = f"🟡 Cooldown {int(s.rate_limited_until - now)}s (#{s.rate_limit_count})"
        else:
            status = "🟢 Ready"
        is_last = cfg.provider == _LAST_USED_PROVIDER
        rows.append({
            "Provider" : ("⭐ " if is_last else "") + cfg.provider,
            "Model"    : cfg.model_id,
            "Cost"     : "FREE",
            "RPD"      : f"{cfg.rpd:,}",
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
    return "No model available"


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
