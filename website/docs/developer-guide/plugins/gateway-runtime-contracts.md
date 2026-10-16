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
