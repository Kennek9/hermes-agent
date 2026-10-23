"""Process-local receipt clocks. No platform I/O, message bodies or identity maps."""

from dataclasses import dataclass
from hashlib import sha256
import math
import os
import time
from uuid import uuid4


_DOMAIN = uuid4().hex
_MAX_PREFIX = 32
_MAX_RECEIPTS = 128


def reading():
    return {
        "domain": _DOMAIN,
        "process_id": os.getpid(),
        "wall": time.time(),
        "tick": time.monotonic(),
    }


def _valid(value):
    return (
        isinstance(value, dict)
        and value.get("domain") == _DOMAIN
        and value.get("process_id") == os.getpid()
        and all(
            isinstance(value.get(k), (int, float))
            and not isinstance(value[k], bool)
            and math.isfinite(value[k])
            for k in ("wall", "tick")
        )
    )


def receive(event):
    """Stamp the live adapter seam once; queue admission must not restart the clock."""
    if getattr(event, "_execution_observation", None):
        invalidate(event)
        return
    if not getattr(event, "_execution_clock_started", False):
        event._execution_clock_started = True
        event._execution_received_clock = None if event.internal else reading()


def invalidate(event):
    """Merged or redirected inputs no longer have a unique inbound measurement."""
    event._execution_clock_started = True
    event._execution_received_clock = None


def _fingerprint(text):
    return sha256(text.encode()).hexdigest()


def _split_receipt(result):
    raw = getattr(result, "raw_response", None)
    return bool(getattr(result, "continuation_message_ids", None)) or (
        isinstance(raw, dict)
        and (
            raw.get("partial_overflow")
            or isinstance(raw.get("message_ids"), (list, tuple))
            and len(raw["message_ids"]) > 1
        )
    )


@dataclass(frozen=True)
class _Receipt:
    prefix_hash: str
    prefix_length: int
    content_hash: str
    content_length: int
    clock: dict


class TextClocks:
    """Keep bounded receipt metadata for final-text reconciliation."""

    def __init__(self):
        self._receipts = []
        self._last = None
        self._overflowed = False
        self._prior_segment = None

    def start_segment(self, visible_prefix=""):
        """Exclude earlier narration at structured tool boundaries."""
        self._receipts.clear()
        self._last = None
        self._overflowed = False
        self._prior_segment = (
            (_fingerprint(visible_prefix), len(visible_prefix))
            if visible_prefix
            else None
        )

    def record(self, text, result, *, native=False):
        accepted = (
            result is True if native else getattr(result, "success", None) is True
        )
        if not accepted:
            return
        raw = getattr(result, "raw_response", None)
        if isinstance(raw, dict) and raw.get("skipped"):
            return
        text = text.strip()
        if self._prior_segment:
            digest, length = self._prior_segment
            if len(text) >= length and _fingerprint(text[:length]) == digest:
                text = text[length:].lstrip()
        if not text or self._overflowed:
            return
        prefix = text[:_MAX_PREFIX]
        key = (_fingerprint(prefix), len(prefix), _fingerprint(text), len(text))
        if (
            self._last is None
            or self._last.content_hash != key[2]
            or self._last.content_length != key[3]
        ):
            self._last = _Receipt(*key, reading())
        # A composite adapter receipt does not expose the first individual ACK.
        if _split_receipt(result):
            return
        if any((r.prefix_hash, r.prefix_length) == key[:2] for r in self._receipts):
            return
        if len(self._receipts) == _MAX_RECEIPTS:
            self._receipts.clear()
            self._overflowed = True
            return
        self._receipts.append(self._last)

    def match(self, text, *, exact=False):
        """Only ACKed characters of the completed answer define first text."""
        text = text.strip()
        if not text or self._overflowed:
            return None
        candidates = self._receipts
        if exact and self._last is not None:
            candidates = [*candidates, self._last]
        for receipt in candidates:
            if exact:
                matches = (
                    len(text) == receipt.content_length
                    and _fingerprint(text) == receipt.content_hash
                )
            else:
                matches = (
                    len(text) >= receipt.prefix_length
                    and _fingerprint(text[: receipt.prefix_length])
                    == receipt.prefix_hash
                )
            if matches:
                return dict(receipt.clock)
        return None


def stream_receipts(consumer, text, result=None):
    """Legacy consumers keep working without claiming clocks they do not implement."""
    if not isinstance(getattr(consumer, "_text_clocks", None), TextClocks):
        return {}
    if result is not None:
        consumer._observe_text_receipt(text, result)
    return {
        "execution_first_text_clock": consumer.text_receipt_clock(text),
        "execution_final_text_clock": consumer.text_receipt_clock(text, exact=True),
    }


def text_receipt(event, text, result):
    """Capture immediately after the real final send, before ledger and cleanup work."""
    if text.strip() and getattr(result, "success", None) is True:
        raw = getattr(result, "raw_response", None)
        if not isinstance(raw, dict) or not raw.get("skipped"):
            event._execution_final_clock = reading()
            event._execution_first_final_clock = (
                None if _split_receipt(result) else event._execution_final_clock
            )


def timings(event, *, confirmed):
    """Require one process clock and keep missing observations unknown."""
    start = getattr(event, "_execution_received_clock", None)
    identity = getattr(event, "_execution_observation", {})
    if (
        not _valid(start)
        or identity.get("generation_failed")
        or getattr(event, "reply_anchor_override", None)
    ):
        return {}
    observed = reading()
    if observed["wall"] < start["wall"] or observed["tick"] < start["tick"]:
        return {}
    result = {
        "received_at": start["wall"],
        "observed_at": observed["wall"],
        "observed_elapsed_ms": (observed["tick"] - start["tick"]) * 1000,
    }
    final = getattr(event, "_execution_final_clock", None)
    candidates = [
        identity.get("first_text_clock"),
        getattr(event, "_execution_first_final_clock", None),
    ]
    valid = [
        v
        for v in candidates
        if _valid(v) and v["tick"] >= start["tick"] and v["wall"] >= start["wall"]
    ]
    if valid:
        first = min(valid, key=lambda v: v["tick"])
        result.update(
            first_text_at=first["wall"],
            first_text_elapsed_ms=(first["tick"] - start["tick"]) * 1000,
        )
    delivery = final if _valid(final) else identity.get("final_text_clock")
    if (
        confirmed
        and _valid(delivery)
        and delivery["tick"] >= start["tick"]
        and delivery["wall"] >= start["wall"]
    ):
        result.update(
            delivered_at=delivery["wall"],
            delivery_elapsed_ms=(delivery["tick"] - start["tick"]) * 1000,
        )
    return result
