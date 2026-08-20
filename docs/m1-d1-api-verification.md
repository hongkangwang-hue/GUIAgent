# M1 D1：多模态 API 连通性验证

> 采集时间：2026-08-20 22:13
> 测试图片：outputs/gallery/smoke-20260819-222534.raw.png

验证两件事：① 模型能否看懂真实桌面截图并输出可解析的结构化结果；② `ChatOpenAI` + 平台 OpenAI 兼容 `base_url` 能否正确传图。

## 结果汇总

| 平台 | 模型 | 状态 | 可解析 | 坐标在范围内 | 平均延迟 | 平均 tokens | 限流重试 |
|---|---|---|---|---|---|---|---|
| 阿里云百炼 | qwen3-vl-8b-instruct | 通过 | 3/3 | 3/3 | 6256ms | 1024 | 0 次 |
| 智谱开放平台 | glm-4.6v-flash | 通过（有限流） | 1/3 | 1/3 | 5556ms | 1272 | 2 次 |

## 逐家详情

### 阿里云百炼

- 模型：`qwen3-vl-8b-instruct`
- 端点：`https://dashscope.aliyuncs.com/compatible-mode/v1`
- 送出图片：{'sent_width': 1024, 'sent_height': 768, 'jpeg_quality': 85, 'downscale': 1.0, 'bytes': 75965}
- 累计用量：{'model': 'qwen3-vl-8b-instruct', 'requests': 3, 'prompt_tokens': 2871, 'completion_tokens': 201, 'cached_tokens': 0, 'total_tokens': 3072, 'cost_cny': 0.0, 'priced': False}

> 已知限制：OpenAI 兼容模式。Qwen-VL 系列原生支持 grounding，是模式 A 的主力

**第 1 次** — left_click {'x': 319, 'y': 48} | 范围内 | 10514ms | 1010 tokens

```
{"action": "left_click", "x": [319, 48], "thinking": "窗口标题栏位于屏幕最上方，包含“GUIAgent 环境搭建与验证”字样，是可点击的区域。"}
```

**第 2 次** — left_click {'x': 319, 'y': 48} | 范围内 | 3928ms | 1015 tokens

```
{"action": "left_click", "x": [319, 48], "thinking": "窗口标题栏位于屏幕最上方，包含“GUIAgent 环境搭建与验证”字样和关闭按钮等控件，符合子任务目标。"}
```

**第 3 次** — left_click {'x': 319, 'y': 48} | 范围内 | 4326ms | 1047 tokens

```
{"action": "left_click", "x": [319, 48], "thinking": "窗口标题栏通常位于屏幕最上方，包含窗口名称和控制按钮（如最小化、最大化、关闭）。在提供的截图中，可以看到一个带有“GUIAgent 环境搭建与验证”文字的标题栏，这符合窗口标题栏的特征。因此，点击该区域可以实现目标。"}
```

### 智谱开放平台

- 模型：`glm-4.6v-flash`
- 端点：`https://open.bigmodel.cn/api/paas/v4`
- 送出图片：{'sent_width': 1024, 'sent_height': 768, 'jpeg_quality': 85, 'downscale': 1.0, 'bytes': 75965}
- 累计用量：{'model': 'glm-4.6v-flash', 'requests': 1, 'prompt_tokens': 1178, 'completion_tokens': 94, 'cached_tokens': 0, 'total_tokens': 1272, 'cost_cny': 0.0, 'priced': False}

> 已知限制：flash 档为免费或低价模型，适合做横评的成本对照组

**第 1 次** — 失败（transient）：智谱开放平台 调用失败（transient）：Error code: 429 - {'error': {'code': '1305', 'message': '该模型当前访问量过大

**第 2 次** — 失败（transient）：智谱开放平台 调用失败（transient）：Error code: 429 - {'error': {'code': '1305', 'message': '该模型当前访问量过大

**第 3 次** — left_click {'x': 259, 'y': 44} | 范围内 | 5556ms | 1272 tokens

```

{"action": "left_click", "x": 259, "y": 44}
```

## 结论

此文件由 `python scripts/verify_llm.py` 生成，是 M1 D1 验收项的证据。
M2 的 `OpenAICompatBackend` 使用的正是这里验证过的消息构造方式，
三家平台共用同一实现，M3 横评时只切换配置。
