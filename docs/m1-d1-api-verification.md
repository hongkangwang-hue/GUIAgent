# M1 D1：多模态 API 连通性验证

> 采集时间：2026-08-24 10:49
> 测试图片：现场截屏 1920×1080

验证两件事：① 模型能否看懂真实桌面截图并输出可解析的结构化结果；② `ChatOpenAI` + 平台 OpenAI 兼容 `base_url` 能否正确传图。

## 结果汇总

| 平台 | 模型 | 状态 | 可解析 | 坐标在范围内 | 平均延迟 | 平均 tokens | 限流重试 |
|---|---|---|---|---|---|---|---|
| 阿里云百炼 | qwen3-vl-8b-instruct | 通过 | 3/3 | 3/3 | 6360ms | 1013 | 0 次 |
| 智谱开放平台 | glm-4.6v-flash | 失败 | — | — | — | — | 3 次 |

## 逐家详情

### 阿里云百炼

- 模型：`qwen3-vl-8b-instruct`
- 端点：`https://dashscope.aliyuncs.com/compatible-mode/v1`
- 送出图片：{'sent_width': 1024, 'sent_height': 768, 'jpeg_quality': 85, 'downscale': 1.0, 'bytes': 178777}
- 累计用量：{'model': 'qwen3-vl-8b-instruct', 'requests': 3, 'prompt_tokens': 2871, 'completion_tokens': 167, 'cached_tokens': 0, 'total_tokens': 3038, 'cost_cny': 0.0, 'priced': False}

> 已知限制：OpenAI 兼容模式。Qwen-VL 系列原生支持 grounding，是模式 A 的主力

**第 1 次** — left_click {'x': 219, 'y': 307} | 范围内 | 8858ms | 1044 tokens

```
{"action": "left_click", "x": [219, 307], "thinking": "窗口标题栏或菜单栏通常位于窗口的最上方。在当前截图中，Windows PowerShell 窗口的标题栏位于其顶部，包含窗口控制按钮（最小化、最大化、关闭），并且有“Windows PowerShell”文字标识。因此，点击该区域可以实现目标。"}
```

**第 2 次** — left_click {'x': 219, 'y': 310} | 范围内 | 4740ms | 997 tokens

```
{"action": "left_click", "x": [219, 310], "thinking": "窗口标题栏位于屏幕最上方，是可点击的区域。"}
```

**第 3 次** — left_click {'x': 219, 'y': 310} | 范围内 | 5483ms | 997 tokens

```
{"action": "left_click", "x": [219, 310], "thinking": "窗口标题栏位于屏幕最上方，是可点击的区域。"}
```

### 智谱开放平台

- 模型：`glm-4.6v-flash`
- 端点：`https://open.bigmodel.cn/api/paas/v4`
- 送出图片：{'sent_width': 1024, 'sent_height': 768, 'jpeg_quality': 85, 'downscale': 1.0, 'bytes': 178777}
- 累计用量：{'model': 'glm-4.6v-flash', 'requests': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'cached_tokens': 0, 'total_tokens': 0, 'cost_cny': 0.0, 'priced': True}

> 已知限制：flash 档为免费或低价模型，适合做横评的成本对照组

**第 1 次** — 失败（transient）：智谱开放平台 调用失败（transient）：Error code: 429 - {'error': {'code': '1305', 'message': '该模型当前访问量过大

**第 2 次** — 失败（transient）：智谱开放平台 调用失败（transient）：Error code: 429 - {'error': {'code': '1305', 'message': '该模型当前访问量过大

**第 3 次** — 失败（transient）：智谱开放平台 调用失败（transient）：Error code: 429 - {'error': {'code': '1305', 'message': '该模型当前访问量过大

## 结论

此文件由 `python scripts/verify_llm.py` 生成，是 M1 D1 验收项的证据。
M2 的 `OpenAICompatBackend` 使用的正是这里验证过的消息构造方式，
三家平台共用同一实现，M3 横评时只切换配置。
