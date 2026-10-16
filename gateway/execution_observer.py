"""Native-turn correlation for actual text delivery; no platform I/O or retries."""

import asyncio
import logging

logger = logging.getLogger(__name__)


def attach_execution(event, result):
    """Preserve this run, even when a later queued turn begins before cleanup."""
    if (
        not isinstance(result, dict)
        or not result.get("session_id")
        or not result.get("turn_id")
    ):
        return
    event._execution_observation = {
        "session_id": result["session_id"],
        "turn_id": result["turn_id"],
        "generation_failed": result.get("failed") is True,
        "stream_confirmed": result.get("execution_delivery_confirmed") is True
        and not result.get("failed"),
    }


def confirm_streamed_text(consumer, text):
    """Require both the consumer receipt and the exact finalized text, not a preview flag."""
    if consumer is None or not text:
        return False
    match = getattr(consumer, "delivered_final_matches", None)
    try:
        return (
            callable(match)
            and match(text) is True
            and bool(
                getattr(consumer, "final_response_sent", False)
                or getattr(consumer, "final_content_delivered", False)
            )
        )
    except Exception:
        return False


async def observe_delivery(event, result):
    """Only a successful adapter result or confirmed final stream proves delivery."""
    identity = getattr(event, "_execution_observation", None)
    if not isinstance(identity, dict):
        return
    outcome = "unknown"
    if result is not None:
        if getattr(result, "success", None) is True:
            outcome = "confirmed"
        elif getattr(result, "success", None) is False and getattr(
            result, "error_kind", None
        ) in {
            "too_long",
            "bad_format",
            "forbidden",
            "not_found",
            "rate_limited",
        }:
            outcome = "failed"
    elif identity.get("stream_confirmed"):
        outcome = "confirmed"
    try:
        from hermes_cli.lifecycle import invoke_hook

        await asyncio.to_thread(
            invoke_hook,
            "gateway_delivery",
            session_id=identity["session_id"],
            turn_id=identity["turn_id"],
            outcome=outcome,
            generation_failed=identity.get("generation_failed", False),
            message_id=str(getattr(result, "message_id", "") or ""),
        )
    except Exception:
        logger.warning("Execution delivery observation failed")
