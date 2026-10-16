---
title: "Gateway runtime contracts"
description: "Public plugin APIs for trusted context, supervised work and exact Slack routing"
---

# Gateway runtime contracts

Plugins read request identity with `ctx.session_value(name, default="")`. It reads only
the gateway's task-local ContextVars. Unknown, unbound or cleared values never inherit
`HERMES_SESSION_*` process environment values. Keep actor and routing data out of tool
parameters supplied by a model.

`ctx.run_async(awaitable)` uses the host's synchronous-to-async bridge and carries the
calling profile and request context. `ctx.spawn_task(coroutine, name=None)` supervises
background work on the profile's host loop and cancels it when the plugin unloads. A
synchronous tool worker may schedule work after the manager has seen its gateway loop.
No active host loop is an error; the plugin must not create an unsupervised worker.

`await ctx.send_platform_message(platform, chat_id, text, reply_to=None, metadata=None)`
returns `{"ok": bool, "message_id": ...}` or a safe error code. Slack requires
`metadata={"workspace_id": "T...", "blocks": [...]}` and an existing conversation id.
The exact workspace client sends on the gateway loop. Unknown workspaces never fall
back to the primary client. `text` remains the accessible notification fallback and
`reply_to` preserves the original thread.

`await ctx.call_slack_api(workspace_id, method, **params)` returns the SDK response
mapping for read-only methods: `conversations.history`, `conversations.replies`,
`conversations.info`, `conversations.members`, `users.info`, and `chat.getPermalink`.
It has the same profile, workspace and loop boundary. Both operations require the
existing `gateway.platform_actions` capability; they do not bypass operator consent.

`await ctx.set_slack_thread_status(workspace_id, channel_id, thread_ts, status)` sets
or clears progress text with the same capability, profile, workspace and loop checks.
An empty status clears it. The adapter retains its SDK compatibility for both the
Assistant Threads and Agent Sessions status APIs. This mutation has a dedicated method
so the source-read API remains read-only.

Register a message shortcut with
`ctx.register_slack_shortcut_handler(callback_id, callback)`. Its callback is
`async def callback(ack, body, *, client)`. Acknowledge within Slack's deadline. `body`
comes from the verified SDK envelope and `client` belongs to that exact workspace.
Unloaded callbacks are not invoked. Existing action callbacks keep their positional
`(ack, body, action)` contract and may request the exact client as a keyword.

`ctx.register_slack_event_handler(event_type, callback)` observes named SDK events as
well as maintenance events. SDK observation does not grant chat or source access:
business capture still uses the authorized `gateway_platform_event` hook. Commands
registered through `ctx.register_command` remain routable when explicitly declared in
the Slack manifest, even if the automatically generated manifest reached 50 entries.

`ctx.register_tool(..., model_visible=False, deprecated=True)` keeps a compatibility
handler callable by its old name while excluding it from model schemas and tool search.
Visibility does not change profile scope, tool override consent or disposal ownership.

Independent Weixin workers may call
`await gateway.platforms.weixin.send_text_message(chat_id=..., text=..., client_id=...)`.
It sends literal text, restores the profile account's peer context token and preserves
chunk idempotency. It never interprets source text as file or MEDIA instructions.

For an explicit business mutation, use
`await ctx.request_confirmation(summary, payload, locale="en-US", timeout_seconds=120)`.
The host reads requester, profile, chat, message and session from trusted ContextVars.
Feishu presents a native interactive card. Weixin presents `/approve <nonce>` and
`/deny <nonce>` through the existing native control-command path. Only the exact
requester and conversation may answer. The host binds one nonce to the payload digest,
expires it and consumes it once. A restart invalidates outstanding confirmations.
Execution YOLO, smart approval, session grants and permanent grants do not apply.
The result contains a host-generated `request_id`, `payload_digest`, trusted `context`
and `status`; only `status="approved"` permits the caller to revalidate its own current
business authorization. Denial, timeout, delivery failure and changed payloads never
authorize a write.

Callers that own a private connector policy can use
`tools.mcp_isolated.IsolatedMCPClient(name, stderr=None, follow_redirects=False)`.
Its public lifecycle is `await start(config)`, `discovered_tools()`,
`await call_tool(name, arguments, timeout=...)`, and `await close()`.
The connection never publishes schemas into the user's registry, never replaces its
explicit endpoint from profile config, and never automatically replays a dispatched
tool call. The caller still owns its tool allowlist, argument validation and write
authorization. Provider stderr is discarded by default.
