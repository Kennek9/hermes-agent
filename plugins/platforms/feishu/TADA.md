# Tada 飞书接入维护

此目录负责飞书 SDK 接入、消息解析、回复准入和标准事件转发。公司知识查询、数据库、微信采集和会议处理由 [ken-agent-V2](https://github.com/Kennek9/ken-agent-V2) 的插件及知识服务负责，不应继续写入适配器。

## 修改入口

| 需要修改的行为 | 代码入口 | 关键约束 |
| --- | --- | --- |
| 群里何时回复 | `adapter.py` 的 `_admit`、`_mentions_self` | 开启 `require_mention` 时，@所有人、@其他人和普通群消息不触发；单独 @机器人触发；同时 @所有人和 @机器人仍算直接提及 |
| 旧版 Channel 消息 | `adapter.py` 的 `_on_legacy_message_event` | `is_mention` 不能单独证明 @机器人；结合原文和服务端去除机器人提及后的正文判断。缺少可确认的提及信息时不回复 |
| 消息入库字段、时间、事件顺序 | `chat_evidence.py` | 不下载附件、不调用模型；用户身份优先 `user_id`，再取 `open_id`；私聊文档先发成员事件，再发消息事件 |
| 消息观察与回复分流 | `adapter.py` 的 `_handle_message_event_data` | 去重后先观察消息，再执行回复准入；不回复不代表不采集 |
| 企业会议结束事件 | `adapter.py` 的 `_on_company_meeting_ended` | 只转交已认证 SDK 事件；需配套公司会议插件持久化，失败不能假装已接收 |

提及的 ID 匹配优先级统一使用 `_FeishuBotIdentity.matches`：双方有 `open_id` 时先比较；否则比较 `user_id`；无法比较 ID 时才按名称匹配。不要增加第二套名称匹配规则。

旧版 SDK 对 `text_without_at_bot` 的定义见 [官方事件模型](https://github.com/larksuite/oapi-sdk-go/blob/v3_main/service/im/v1/ext_model.go)：它移除对当前机器人的提及，只有 @机器人时可能省略该字段。保留这些差异的回归用例，不通过宽泛的 `is_mention` 恢复群聊放行。

## 验证与升级

使用独立工作树和独立测试环境，运行 `scripts/run_tests.sh tests/gateway/test_feishu*.py`。其中 `test_feishu_direct_mentions.py` 验证回复边界，`test_feishu_chat_evidence.py` 验证纯事件构造，`test_feishu_storage_events.py` 验证实际适配器转发。

此分支保留了 Tada 服务器已有的文件链接采集、`parent_message_id` 和企业会议回调。升级前核对服务器未提交差异；不能仅按 GitHub 默认分支覆盖运行目录。新建的 `chat_evidence.py` 必须与 `adapter.py` 同时交付，否则适配器不能导入。不要对运行中的服务执行全量代码覆盖或使用生产用户目录运行测试。
