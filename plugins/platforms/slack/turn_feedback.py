"""One inbound message owns its acknowledgement and optional in-thread progress post."""

import asyncio
import logging

from gateway.execution_clock import reading

logger = logging.getLogger(__name__)


class TurnFeedback:
    def __init__(self, adapter, team, channel, root, message, ingress, chinese):
        self.adapter, self.team, self.channel = adapter, team, channel
        self.root, self.message, self.ingress = root, message, ingress
        self.chinese = chinese
        self.ack = self.posted = None
        self.finalized = False
        self.stop = asyncio.Event()
        self.closing = None
        self.loop = asyncio.get_running_loop()
        self.eyes = asyncio.create_task(self._eyes())
        self.progress = asyncio.create_task(self._progress())

    async def _eyes(self):
        if self.adapter._reactions_enabled() and await self.adapter._react(
            self.channel, self.message, "eyes", self.team, remove=False
        ):
            self.ack = reading()

    async def _progress(self):
        try:
            remaining = max(0, 2.5 - (reading()["tick"] - self.ingress["tick"]))
            await asyncio.wait_for(self.stop.wait(), timeout=remaining)
            return
        except asyncio.TimeoutError:
            pass
        if self.stop.is_set() or self.adapter._outbound_blocked(
            self.channel, "progress"
        ):
            return
        try:
            response = await self.adapter._get_client(
                self.channel, team_id=self.team
            ).chat_postMessage(
                channel=self.channel,
                thread_ts=self.root,
                text="正在处理…" if self.chinese else "Working on it…",
                unfurl_links=False,
                unfurl_media=False,
            )
            if response.get("ok") and response.get("ts"):
                self.posted = response["ts"]
        except Exception:
            logger.debug("Slack progress unavailable", exc_info=True)

    async def finish_progress(self):
        self.stop.set()
        await self.progress
        return self.posted

    def discard(self):
        # Queued inputs can be coalesced or redirected and lose their original owner.
        if not self.loop.is_closed():
            self.loop.call_soon_threadsafe(lambda: self.loop.create_task(self.close()))

    async def close(self):
        if self.closing is None:
            self.closing = asyncio.create_task(self._close())
        await self.closing

    async def _close(self):
        key = (self.team, self.channel, self.message)
        registry = getattr(self.adapter, "_turn_feedback", {})
        if registry.get(key) is self:
            registry.pop(key)
        await self.finish_progress()
        await self.eyes
        if self.ack:
            await self.adapter._react(
                self.channel, self.message, "eyes", self.team, remove=True
            )
        if self.posted and not self.finalized:
            try:
                await self.adapter._get_client(
                    self.channel, team_id=self.team
                ).chat_delete(channel=self.channel, ts=self.posted)
            except Exception:
                logger.debug("Slack progress cleanup unavailable", exc_info=True)
