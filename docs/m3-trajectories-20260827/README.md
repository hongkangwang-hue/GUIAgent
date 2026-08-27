# 端到端轨迹（2026-08-27，档 A）

对应 `docs/m3-微调效果对比分析报告.md` §11.7。

## 来源

客机（VMware Windows 11，1920×1080）跑 `run_basic_tasks.py --execute`，
25 轮（5 任务 × 5 次），tag `lora-fixed-2`，模板 `executor_v0`，
后端为宿主机的 `serve_local_model.py` + adapter `20260827-174327`。

聚合结果见 `docs/m2-runs/20260827-163522-all-exec-offline-lora-fixed-2.json`。

## 只导结构化数据，不导 `frames/`

每个目录只有 `meta.json` 与 `steps.jsonl`。**`frames/` 下的逐步截图
留在客机、未导出。**

理由不是体积，是安全：本项目记录过三次真实事故——Agent 改写并保存过
`.env`、在真实微软账号登录框按了 12 次回车、走到过勾着「保存的密码」的
Edge 数据导入对话框。**轨迹截图会把这类内容带出隔离环境**，
所以结构化导出是默认，截图需要时单独评估。

提交前对这 25 轮做过扫描：

```
model_thinking 非空 146 条（去重 59 条），全部与任务相关
API Key / 邮箱 / 密码 / .env / 登录 等模式    未命中
```

## 字段说明

`steps.jsonl` 每行一步，关键字段：

| 字段 | 含义 |
|---|---|
| `raw_output` | 模型原始输出（JSON 字符串） |
| `model_thinking` | 模型给的理由 —— §11.7.6 的主要证据 |
| `action_model_coords` | 模型坐标空间（0~1000）里的动作 |
| `action_real_coords` | 映射到客机 1920×1080 之后的实际坐标 |
| `execution_status` | `ok` = 动作执行成功；**不代表任务成功** |
| `screenshot_before/after` | 指向 `frames/` 的相对路径，**本导出中不存在** |

> `execution_status = ok` 与任务成功是两回事，这正是 §11.7.6 的要点：
> `close_app` 五轮全部 `ok`，计算器五次都没关掉。
