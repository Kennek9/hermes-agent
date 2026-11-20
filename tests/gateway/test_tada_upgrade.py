"""Tada contracts through the installed Hermes execution and plugin interfaces."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


def test_installed_plugins_register_with_real_manager(tmp_path, monkeypatch, tada_release_plugin_parent):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.syspath_prepend(str(tada_release_plugin_parent))
    (tmp_path / 'plugins').mkdir()
    for name in ('company_knowledge_v1', 'table_workflows'):
        (tmp_path / 'plugins' / name).symlink_to(tada_release_plugin_parent / name)
    (tmp_path / 'config.yaml').write_text('plugins:\n  enabled: [company_knowledge_v1, table_workflows]\n')
    from hermes_cli.plugins import PluginManager
    from tools.registry import registry

    manager = PluginManager(scope_key=str(tmp_path))
    manager.discover_and_load()
    for name in ('company_knowledge_v1', 'table_workflows'):
        plugin = manager._plugins.get(name)
        assert plugin is not None, name
        assert plugin.enabled and not plugin.error, (name, plugin.error)
    assert {
        'tada_knowledge_search', 'knowledge.read_document', 'conversation.memory',
        'workspace.continuity', 'tables.workflow',
    } <= manager._plugin_tool_names
    assert 'knowledge.ask' not in manager._plugin_tool_names
    search = registry.get_entry('tada_knowledge_search', scope=manager.scope_key)
    assert search is not None and search.model_visible
    assert {'query', 'intent'} <= set(search.schema['parameters']['required'])
    assert 'source' not in search.schema['parameters']['properties']
    assert registry.get_entry('knowledge.ask', scope=manager.scope_key) is None
    assert {'gateway_delivery','context_handoff','gateway_platform_event'} <= manager._hooks.keys()


@pytest.mark.parametrize('status', [None,400,429])
def test_credit_exhaustion_is_terminal_and_reported_once(status):
    from agent.error_classifier import classify_api_error
    from gateway.run import _sanitize_gateway_final_response, _prepare_gateway_status_message
    from gateway.config import Platform
    error = RuntimeError('You have no credits remaining. Add credits to continue.')
    error.status_code = status
    result = classify_api_error(error, provider='custom')
    assert result.reason.value == 'billing' and not result.retryable
    text = 'Billing or credits exhausted: ' + str(error)
    assert '额度已用尽' in _sanitize_gateway_final_response(Platform.FEISHU, text)
    assert _prepare_gateway_status_message(Platform.FEISHU, 'error', text) is None


def test_native_transport_maps_tool_result_back_to_public_name():
    from agent.transports.chat_completions import ChatCompletionsTransport
    transport = ChatCompletionsTransport()
    kw = transport.build_kwargs(
        model='gpt-4.1', messages=[{'role':'user','content':'Find policy'}],
        tools=[{'type':'function','function':{'name':'knowledge.ask','description':'Read knowledge','parameters':{'type':'object'}}}],
    )
    wire = kw['tools'][0]['function']['name']
    assert '.' not in wire
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content=None, tool_calls=[SimpleNamespace(id='c1',function=SimpleNamespace(name=wire,arguments='{}'))]
    ),finish_reason='tool_calls')],usage=None)
    assert transport.normalize_response(response).tool_calls[0].name == 'knowledge.ask'


@pytest.mark.asyncio
@pytest.mark.parametrize('locale', ['zh-CN', 'en-US'])
async def test_busy_weixin_message_queues_without_retired_intake(
    tmp_path, monkeypatch, tada_release_plugin_parent, locale,
):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('TADA_DEFAULT_LOCALE', locale)
    monkeypatch.setenv('TADA_WEIXIN_COMPANY_SHARED', 'true')
    monkeypatch.setenv('TADA_WEIXIN_EMPLOYEE_ID', 'employee')
    monkeypatch.syspath_prepend(str(tada_release_plugin_parent))
    (tmp_path / 'plugins').mkdir()
    (tmp_path / 'plugins' / 'company_knowledge_v1').symlink_to(tada_release_plugin_parent / 'company_knowledge_v1')
    (tmp_path / 'config.yaml').write_text('plugins:\n  enabled: [company_knowledge_v1]\n')
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.event import MessageEvent, MessageType
    from gateway.platforms.weixin import WeixinAdapter
    from gateway.run import GatewayRunner
    from hermes_cli.plugins import get_plugin_manager

    manager = get_plugin_manager()
    manager.discover_and_load()
    plugin = manager._plugins['company_knowledge_v1']
    assert plugin.enabled and not plugin.error, plugin.error
    assert 'pre_gateway_dispatch' in manager._hooks

    adapter = WeixinAdapter(PlatformConfig(enabled=True, extra={
        'dm_policy': 'allowlist', 'allow_from': ['employee'],
    }))
    send = AsyncMock()
    monkeypatch.setattr(adapter, '_send_with_retry', send)
    start_turn = AsyncMock()
    adapter.set_message_handler(start_turn)
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.adapters = {Platform.WEIXIN: adapter}
    runner._busy_input_mode = runner._busy_text_mode = 'queue'
    runner._draining = False
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)

    cache = tmp_path / 'cache'
    cache.mkdir()
    attachment = cache / 'busy.txt'
    attachment.write_text('Upgrade attachment acceptance')
    event = MessageEvent(
        text='', message_type=MessageType.TEXT,
        source=adapter.build_source(chat_id='peer', user_id='employee', chat_type='dm'),
        message_id='busy-acceptance', media_urls=[str(attachment)], media_types=['text/plain'],
    )
    key = adapter._event_session_key(event)
    guard = asyncio.Event()
    adapter._active_sessions[key] = guard

    await adapter.handle_message(event)

    assert event._bot_loop_admitted and event._gateway_accepted
    assert adapter._pending_messages[key] is event
    assert event.media_urls == [str(attachment)]
    assert adapter._active_sessions[key] is guard and not guard.is_set()
    start_turn.assert_not_awaited()
    send.assert_not_awaited()
    assert not (tmp_path / 'weixin-jobs.sqlite').exists()
    assert not (tmp_path / 'weixin_uploads').exists()
    assert attachment.read_text() == 'Upgrade attachment acceptance'


def test_context_notice_survives_turn_cleanup_but_resets_with_conversation(monkeypatch):
    from gateway.run import GatewayRunner
    from gateway.session_state import ConversationState
    from gateway.config import Platform
    import gateway.run
    monkeypatch.setattr(gateway.run, '_load_gateway_config', lambda: {})
    runner = object.__new__(GatewayRunner)
    state = ConversationState()
    result = {'last_prompt_tokens': 100000, 'context_length': 200000, 'model': 'test'}
    source = SimpleNamespace(platform=Platform.FEISHU)
    assert '/reset' in runner._hmwa_runtime_footer_line(result, source, 1, state)
    assert runner._hmwa_runtime_footer_line(result, source, 1, state) == ''
    state.clear()
    assert '/reset' in runner._hmwa_runtime_footer_line(result, source, 1, state)
