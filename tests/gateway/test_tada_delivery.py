"""Run inside the patched Hermes test checkout, using its real platform dispatch."""

from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.execution_observer import attach_execution
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.session import SessionSource, build_session_key


class Adapter(BasePlatformAdapter):
    def __init__(self, result):
        super().__init__(
            PlatformConfig(enabled=True, token="synthetic"), Platform.SLACK
        )
        self.result = result
        self.sent = []

    async def connect(self):
        return True

    async def disconnect(self):
        return None

    async def get_chat_info(self, chat_id):
        return {}

    async def send_typing(self, chat_id, metadata=None):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        return self.result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("success", "error_kind", "expected"),
    [
        (True, None, "confirmed"),
        (False, "forbidden", "failed"),
        (False, "unknown", "unknown"),
        (False, "transient", "unknown"),
    ],
)
async def test_real_platform_dispatch_reports_only_its_native_receipt(
    monkeypatch, success, error_kind, expected
):
    import hermes_cli.lifecycle

    calls = []
    monkeypatch.setattr(
        hermes_cli.lifecycle,
        "invoke_hook",
        lambda name, **args: calls.append((name, args)),
    )
    adapter = Adapter(
        SendResult(
            success=success,
            message_id="synthetic-platform-message" if success else None,
            error=None if success else "synthetic rejection",
            error_kind=error_kind,
        )
    )

    async def handler(event):
        attach_execution(
            event, {"session_id": "actual-session", "turn_id": "actual-turn"}
        )
        return "Native dispatch acceptance"

    adapter.set_message_handler(handler)
    source = SessionSource(
        platform=Platform.SLACK, user_id="U_LOCAL", chat_id="D_LOCAL", chat_type="dm"
    )
    event = MessageEvent(
        text="Synthetic request", message_type=MessageType.TEXT, source=source
    )
    await adapter._process_message_background(event, build_session_key(source))
    observed = [value for name, value in calls if name == "gateway_delivery"]
    assert adapter.sent[0] == "Native dispatch acceptance"
    # The existing native sender makes one plain-text fallback after rejection.
    # Observation must preserve that behavior and must not add another send.
    assert len(adapter.sent) == (1 if success else 2)
    assert len(observed) == 1
    assert observed[0]["session_id"] == "actual-session"
    assert observed[0]["turn_id"] == "actual-turn"
    assert observed[0]["outcome"] == expected


@pytest.mark.asyncio
async def test_stream_preview_flag_alone_is_not_a_receipt(monkeypatch):
    import hermes_cli.lifecycle
    from gateway.execution_observer import observe_delivery, confirm_streamed_text

    calls = []
    monkeypatch.setattr(
        hermes_cli.lifecycle, "invoke_hook", lambda name, **args: calls.append(args)
    )
    event = SimpleNamespace()
    attach_execution(event, {"session_id": "s", "turn_id": "t", "already_sent": True})
    await observe_delivery(event, None)
    assert calls[0]["outcome"] == "unknown"
    consumer = SimpleNamespace(
        final_response_sent=True,
        delivered_final_matches=lambda text: text == "complete",
    )
    assert confirm_streamed_text(consumer, "preview") is False
    assert confirm_streamed_text(consumer, "complete") is True
