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
    manager = PluginManager(scope_key=str(tmp_path))
    manager.discover_and_load()
    for name in ('company_knowledge_v1', 'table_workflows'):
        plugin = manager._plugins.get(name)
        assert plugin is not None, name
        assert plugin.enabled and not plugin.error, (name, plugin.error)
    assert {'knowledge.ask','knowledge.read_document','conversation.memory','workspace.continuity','tables.workflow'} <= manager._plugin_tool_names
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
@pytest.mark.parametrize('locale,receipt', [
    ('zh-CN', '已收到，等待后台处理'),
    ('en-US', 'Received. Waiting for background processing.'),
])
async def test_busy_weixin_intake_persists_before_agent_routing(tmp_path, monkeypatch, tada_release_plugin_parent, locale, receipt):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('TADA_WEIXIN_COMPANY_SHARED', 'true')
    monkeypatch.setenv('TADA_WEIXIN_EMPLOYEE_ID', 'employee')
    monkeypatch.syspath_prepend(str(tada_release_plugin_parent))
    from company_knowledge_v1 import weixin, weixin_jobs, settings
    settings.configure(SimpleNamespace(get_config=lambda key, default=None: locale if key == 'default_locale' else default))
    import hermes_cli.lifecycle
    from gateway.config import Platform
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource
    from gateway.run_busy import GatewayBusySessionMixin
    cache = tmp_path / 'cache'; cache.mkdir()
    attachment=cache/'busy.txt';attachment.write_text('Upgrade attachment acceptance')
    event=MessageEvent(text='',message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.WEIXIN,chat_id='peer',user_id='employee',chat_type='dm'),
        message_id='busy-acceptance',media_urls=[str(attachment)],media_types=['text/plain'])
    adapter=SimpleNamespace(_send_with_retry=AsyncMock())
    runner=SimpleNamespace(_is_user_authorized_for_source=lambda s:True,
        _admit_bot_message_for_source=lambda s:True,_delivery_adapter_for=lambda s:adapter)
    monkeypatch.setattr(hermes_cli.lifecycle,'invoke_hook',lambda name,**kw:[weixin.pre_gateway_dispatch(**kw)])
    assert await GatewayBusySessionMixin._handle_active_session_busy_message(runner,event,'session') is True
    assert adapter._send_with_retry.await_count == 1
    assert adapter._send_with_retry.call_args.kwargs['content'] == receipt
    import sqlite3
    with sqlite3.connect(weixin_jobs.job_database()) as connection:
        assert connection.execute('SELECT count(*) FROM jobs').fetchone()[0] == 1


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
