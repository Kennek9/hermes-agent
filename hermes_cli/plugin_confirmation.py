"""One-use human confirmations for plugin mutations, independent of execution policy."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermes_cli.approval_transport import ApprovalDecision, ApprovalRequest
from hermes_constants import get_hermes_home
from tools.interrupt import (
    acting_for_tid, capture_interrupt_checkpoint, interrupt_checkpoint_changed,
    run_if_interrupt_checkpoint_unchanged,
)

_NONCE = re.compile(r"[A-Za-z0-9_-]{43}")
_LOCK = threading.RLock()
_PENDING: dict[str, "PendingConfirmation"] = {}
_MAX_PENDING = 256
_MAX_PAYLOAD_BYTES = 64_000


def payload_digest(payload: dict) -> str:
    """Bind a complete immutable JSON request rather than a model approval flag."""
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(raw) > _MAX_PAYLOAD_BYTES:
        raise ValueError("confirmation_payload_too_large")
    return hashlib.sha256(raw).hexdigest()


def messages(locale: str) -> dict[str, str]:
    language = "zh-CN" if str(locale).replace("_", "-").split("-", 1)[0].lower() == "zh" else "en-US"
    return json.loads((Path(__file__).parent / "data/plugin_confirmation_messages.json").read_text(encoding="utf-8"))[language]


@dataclass(frozen=True)
class ConfirmationContext:
    profile_home: str
    platform: str
    user_id: str
    user_id_alt: str
    chat_id: str
    chat_type: str
    thread_id: str
    message_id: str
    session_key: str

    @classmethod
    def read(cls, ctx) -> "ConfirmationContext":
        read = ctx.session_value
        value = cls(
            profile_home=str(get_hermes_home().resolve()),
            platform=read("HERMES_SESSION_PLATFORM"), user_id=read("HERMES_SESSION_USER_ID"),
            user_id_alt=read("HERMES_SESSION_USER_ID_ALT"), chat_id=read("HERMES_SESSION_CHAT_ID"),
            chat_type=read("HERMES_SESSION_CHAT_TYPE"), thread_id=read("HERMES_SESSION_THREAD_ID"),
            message_id=read("HERMES_SESSION_MESSAGE_ID"), session_key=read("HERMES_SESSION_KEY"),
        )
        if (not all((value.user_id, value.chat_id, value.message_id, value.session_key))
                or value.platform not in {"feishu", "weixin"}
                or Path(ctx._manager.scope_key).resolve() != Path(value.profile_home)):
            raise ValueError("trusted_context_missing")
        return value


@dataclass
class PendingConfirmation:
    nonce: str
    plugin_id: str
    context: ConfirmationContext
    request: ApprovalRequest
    payload_digest: str
    deadline: float
    future: asyncio.Future
    adapter: Any
    labels: dict[str, str]
    interrupt_checkpoint: tuple[tuple[int, int], ...]
    card_message_id: str = ""
    consumed: bool = False

    def finish(self, choice: str) -> bool:
        """Consume under the broker lock before delivering a correlated SDK decision."""
        with _LOCK:
            if (self.consumed or self.future.done() or time.monotonic() >= self.deadline
                    or choice not in {"once", "deny"} or _PENDING.get(self.nonce) is not self):
                return False
            def commit() -> None:
                self.consumed = True
                decision = self.request.respond(choice)
                self.future.get_loop().call_soon_threadsafe(_deliver, self.future, decision)
            return run_if_interrupt_checkpoint_unchanged(self.interrupt_checkpoint, commit)


def _deliver(future: asyncio.Future, decision: ApprovalDecision) -> None:
    if not future.done():
        future.set_result(decision)


def _retire(entry: PendingConfirmation) -> None:
    with _LOCK:
        if _PENDING.get(entry.nonce) is entry:
            _PENDING.pop(entry.nonce)
        entry.consumed = True
    if not entry.future.done() and not entry.future.get_loop().is_closed():
        entry.future.get_loop().call_soon_threadsafe(entry.future.cancel)


async def request_confirmation(ctx, summary: str, payload: dict, *, locale: str, timeout_seconds: float) -> dict:
    """Present only native human UI. YOLO, smart approval and allowlists never grant consent."""
    checkpoint = capture_interrupt_checkpoint((threading.get_ident(), acting_for_tid.get()))
    if interrupt_checkpoint_changed(checkpoint):
        return {"status": "cancelled", "error": "confirmation_cancelled"}
    if not isinstance(summary, str) or not summary.strip() or len(summary) > 4000 or not isinstance(payload, dict):
        return {"status": "unavailable", "error": "invalid_confirmation_request"}
    if not isinstance(timeout_seconds, (int, float)) or not 0 < timeout_seconds <= 300:
        return {"status": "unavailable", "error": "invalid_confirmation_timeout"}
    try:
        context = ConfirmationContext.read(ctx)
    except (ValueError, TypeError):
        return {"status": "unavailable", "error": "trusted_context_missing"}
    try:
        digest = payload_digest(payload)
    except (ValueError, TypeError):
        return {"status": "unavailable", "error": "invalid_confirmation_request"}
    adapter, error = ctx.platform_actions._gate(context.platform, chat_id=context.chat_id)
    if error is not None or adapter is None:
        return {"status": "unavailable", "error": (error or {}).get("error", "platform_unavailable")}
    labels = messages(locale)
    if interrupt_checkpoint_changed(checkpoint):
        return {"status": "cancelled", "error": "confirmation_cancelled"}
    approval = ApprovalRequest.create(
        command=summary, description=labels["description"], pattern_key="plugin_confirmation:" + digest,
        pattern_keys=("plugin_confirmation:" + digest,), session_key=context.session_key,
        surface=context.platform, allow_session=False, allow_permanent=False, timeout_seconds=timeout_seconds,
    )

    async def present_and_wait() -> dict:
        entry = PendingConfirmation(
            nonce=secrets.token_urlsafe(32), plugin_id=ctx.plugin_id, context=context,
            request=approval, payload_digest=digest, deadline=time.monotonic() + timeout_seconds,
            future=asyncio.get_running_loop().create_future(), adapter=adapter, labels=labels,
            interrupt_checkpoint=checkpoint,
        )
        with _LOCK:
            if len(_PENDING) >= _MAX_PENDING:
                return {"status": "unavailable", "error": "confirmation_capacity_exhausted"}
            _PENDING[entry.nonce] = entry
        handle = ctx._track("human_confirmation", approval.request_id, lambda: _retire(entry))
        receipt = {
            "request_id": approval.request_id, "payload_digest": digest,
            "context": {key: getattr(context, key) for key in (
                "profile_home", "platform", "user_id", "chat_id", "message_id", "session_key",
            )},
        }
        try:
            sent = await _wait_for_stage(asyncio.create_task(adapter.send_plugin_confirmation(entry)), entry)
            if not sent or not sent.success:
                return {**receipt, "status": "unavailable", "error": "confirmation_delivery_failed"}
            entry.card_message_id = str(sent.message_id or "")
            if context.platform == "feishu" and not entry.card_message_id:
                return {**receipt, "status": "unavailable", "error": "confirmation_delivery_failed"}
            from tools.approval_human_wait import activity_heartbeat, human_wait_window
            heartbeat = activity_heartbeat("waiting for plugin confirmation")
            with human_wait_window(context.session_key):
                decision = await _wait_for_stage(entry.future, entry, heartbeat=heartbeat)
            if (not isinstance(decision, ApprovalDecision) or decision.request_id != approval.request_id
                    or decision.request_digest != approval.digest or decision.choice not in approval.allowed_choices
                    or payload_digest(payload) != digest):
                return {**receipt, "status": "denied", "error": "confirmation_binding_mismatch"}
            return {**receipt, "status": "approved" if decision.choice == "once" else "denied"}
        except _RequestEnded as ended:
            return {**receipt, "status": ended.status, "error": "confirmation_" + ended.status}
        finally:
            handle.dispose()

    result = await ctx.platform_actions._on_gateway_loop(present_and_wait)
    if interrupt_checkpoint_changed(checkpoint):
        return {**result, "status": "cancelled", "error": "confirmation_cancelled"}
    return result if "status" in result else {"status": "unavailable", "error": result.get("error", "platform_unavailable")}


class _RequestEnded(Exception):
    def __init__(self, status: str):
        self.status = status


async def _wait_for_stage(task: asyncio.Future, entry: PendingConfirmation, *, heartbeat=lambda: None):
    """Bound delivery and human wait by one deadline and the original worker's cancellation."""
    try:
        while True:
            if interrupt_checkpoint_changed(entry.interrupt_checkpoint):
                raise _RequestEnded("cancelled")
            remaining = entry.deadline - time.monotonic()
            if remaining <= 0:
                raise _RequestEnded("expired")
            if task.done():
                return task.result()
            await asyncio.wait({task}, timeout=min(0.1, remaining))
            heartbeat()
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


def resolve_feishu_card(adapter, event, value: dict) -> dict[str, str] | None:
    """An SDK-verified card click must match the exact presented card and original actor."""
    nonce, choice = value.get("hermes_plugin_confirmation"), value.get("choice")
    if not isinstance(nonce, str) or not _NONCE.fullmatch(nonce):
        return None
    with _LOCK:
        entry = _PENDING.get(nonce)
        if entry is None or entry.adapter is not adapter or entry.context.platform != "feishu":
            return None
        operator, callback = getattr(event, "operator", None), getattr(event, "context", None)
        identifiers = {str(getattr(operator, field, "") or "") for field in ("user_id", "open_id")}
        if (not adapter._is_interactive_operator_authorized(str(getattr(operator, "open_id", "") or ""))
                or entry.context.user_id not in identifiers
                or str(getattr(callback, "open_chat_id", "") or "") != entry.context.chat_id
                or not entry.card_message_id
                or str(getattr(callback, "open_message_id", "") or "") != entry.card_message_id):
            return None
        if not entry.finish(choice):
            return None
        return {"title": entry.labels["title"], "text": entry.labels["approved" if choice == "once" else "denied"],
                "color": "green" if choice == "once" else "red"}


def resolve_native_command(event) -> str | None:
    """Consume only explicit native /approve or /deny plus a complete opaque nonce."""
    if not event.allow_gateway_control or event.internal or event.source is None:
        return None
    command, nonce = event.get_command(), event.get_command_args().strip()
    if command not in {"approve", "deny"} or not _NONCE.fullmatch(nonce):
        return None
    with _LOCK:
        entry = _PENDING.get(nonce)
        if entry is None:
            return messages("en-US")["expired"] + "\n" + messages("zh-CN")["expired"]
        source = event.source
        if (entry.context.platform != "weixin" or getattr(source.platform, "value", source.platform) != "weixin"
                or entry.context.profile_home != str(get_hermes_home().resolve())
                or source.user_id != entry.context.user_id or source.chat_id != entry.context.chat_id
                or (source.thread_id or "") != entry.context.thread_id):
            return entry.labels["mismatch"]
        choice = "once" if command == "approve" else "deny"
        if not entry.finish(choice):
            return entry.labels["expired"]
        return entry.labels["approved" if choice == "once" else "denied"]
