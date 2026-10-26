from unittest.mock import AsyncMock
import pytest
from gateway.config import PlatformConfig
from hermes_cli import plugins
from plugins.platforms.slack.adapter import SlackAdapter

@pytest.mark.asyncio
async def test_home_workspace_client_and_dispose(monkeypatch):
    manager=plugins.PluginManager()
    ctx=plugins.PluginContext(plugins.PluginManifest(name="team_test",source="user"),manager)
    seen=[]
    async def callback(*,event,body,client): seen.append((event["user"],client))
    handle=ctx.register_slack_event_handler("app_home_opened",callback)
    monkeypatch.setattr(plugins,"get_plugin_manager",lambda:manager)
    adapter=SlackAdapter(PlatformConfig(enabled=False))
    adapter._team_clients={"T_A":object(),"T_B":object()}
    adapter.handle_message=AsyncMock(side_effect=AssertionError("Home is not chat"))
    event={"type":"app_home_opened","tab":"home","user":"U_A"}
    await adapter._handle_app_home_opened(event,{"team_id":"T_B"})
    assert seen==[("U_A",adapter._team_clients["T_B"])]
    await adapter._handle_app_home_opened(event,{"team_id":"UNKNOWN"})
    await adapter._handle_app_home_opened({**event,"team":"T_A"},{"team_id":"T_B"})
    assert len(seen)==1
    handle.dispose()
    await adapter._handle_app_home_opened(event,{"team_id":"T_B"})
    assert len(seen)==1
    adapter.handle_message.assert_not_awaited()

@pytest.mark.asyncio
async def test_action_legacy_names_and_optional_client():
    adapter=SlackAdapter(PlatformConfig(enabled=False))
    adapter._team_clients={"T_B":object()}
    ack=AsyncMock(); seen=[]
    async def legacy(_ack,_payload,_action):
        seen.append("legacy"); await _ack()
    async def current(ack,body,action,*,client):
        seen.append(client); await ack()
    body={"team":{"id":"T_B"}}
    await adapter._invoke_plugin_action_callback(legacy,ack,body,{})
    await adapter._invoke_plugin_action_callback(current,ack,body,{})
    await adapter._invoke_plugin_action_callback(current,ack,{"team":{"id":"UNKNOWN"}},{})
    assert seen==["legacy",adapter._team_clients["T_B"]]
    assert ack.await_count==3

@pytest.mark.asyncio
@pytest.mark.parametrize("remote_file", [False, True])
async def test_ambient_source_arrives_even_when_reply_is_suppressed(remote_file):
    from unittest.mock import MagicMock
    adapter=SlackAdapter(PlatformConfig(enabled=False))
    adapter._bot_user_id="U_BOT"
    adapter._team_bot_user_ids={"T_A":"U_BOT"}
    adapter._app=MagicMock()
    adapter._resolve_user_is_bot=AsyncMock(return_value=False)
    adapter.config.extra.update({"require_mention":True,"strict_mention":True})
    adapter._should_wake_on_unmentioned_message=AsyncMock(return_value=False)
    adapter.handle_message=AsyncMock()
    observed=AsyncMock()
    adapter.set_platform_event_handler(observed)
    event={"type":"message","text":"方案A因成本高放弃","user":"U_A","channel":"C_A",
           "channel_type":"channel","ts":"1790570000.000001","client_msg_id":"synthetic"}
    if remote_file:
        event.update(subtype="file_share", text="", files=[{
            "is_external": True,
            "external_url": "https://docs.google.com/document/d/native-share/edit",
        }])
    await adapter._handle_slack_message(event,{"team_id":"T_A","event_id":"Ev1"})
    assert observed.await_count==1
    envelope,_=observed.await_args.args
    assert envelope["event_type"]=="message_received"
    assert envelope["payload"]["event_kind"]=="created"
    if remote_file:
        assert envelope["payload"]["files"] == event["files"]
        assert envelope["payload"]["content"] == ""
    adapter.handle_message.assert_not_awaited()
