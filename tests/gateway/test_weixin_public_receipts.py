"""Worker receipts use the real iLink HTTP protocol and the selected profile."""

from __future__ import annotations

import pytest
from aiohttp import web

from agent.secret_scope import reset_secret_scope, set_secret_scope
from gateway.platforms import weixin
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.mark.asyncio
async def test_literal_receipts_keep_account_context_chunking_and_stable_ids(tmp_path, monkeypatch):
    requests = []

    async def receive(request):
        requests.append((request.headers["Authorization"], await request.json()))
        return web.json_response({"ret": 0, "errcode": 0})

    app = web.Application()
    app.router.add_post("/ilink/bot/sendmessage", receive)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    endpoint = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
    monkeypatch.setenv("WEIXIN_TOKEN", "foreign-launch-token")
    try:
        for label in ("A", "B", "A"):
            home = tmp_path / label
            home.mkdir(exist_ok=True)
            weixin.ContextTokenStore(str(home))._persist("account-" + label, {"peer": "context-" + label})
            home_token = set_hermes_home_override(home)
            secret_token = set_secret_scope({
                "WEIXIN_ACCOUNT_ID": "account-" + label, "WEIXIN_TOKEN": "token-" + label,
                "WEIXIN_BASE_URL": endpoint,
            }, profile_home=str(home))
            try:
                assert await weixin.send_text_message(
                    chat_id="peer", text="MEDIA:/untrusted-file\n" + "中" * 3800,
                    client_id="worker-receipt",
                )
            finally:
                reset_secret_scope(secret_token)
                reset_hermes_home_override(home_token)
    finally:
        await runner.cleanup()
    assert len(requests) == 6
    for offset, label in ((0, "A"), (2, "B"), (4, "A")):
        chunks = requests[offset:offset + 2]
        assert all(token == "Bearer token-" + label for token, _ in chunks)
        assert all(payload["msg"]["context_token"] == "context-" + label for _, payload in chunks)
        items = [payload["msg"]["item_list"] for _, payload in chunks]
        assert all(len(item) == 1 and item[0]["type"] == weixin.ITEM_TEXT for item in items)
        assert "".join(item[0]["text_item"]["text"] for item in items) == "MEDIA:/untrusted-file\n" + "中" * 3800
    assert requests[0][1]["msg"]["client_id"] == requests[4][1]["msg"]["client_id"]
    assert requests[0][1]["msg"]["client_id"] != requests[1][1]["msg"]["client_id"]


@pytest.mark.asyncio
async def test_missing_scoped_credential_never_uses_launch_token(tmp_path, monkeypatch):
    monkeypatch.setenv("WEIXIN_TOKEN", "foreign-launch-token")
    token = set_secret_scope({"WEIXIN_ACCOUNT_ID": "secondary"}, profile_home=str(tmp_path))
    try:
        assert not await weixin.send_text_message(chat_id="peer", text="Receipt", client_id="id")
    finally:
        reset_secret_scope(token)
