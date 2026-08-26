# GUI Agent 四个数据集方案（最终版）

> 2026-08-26 修订。本版每个数字都在本机数据上实跑过，口径见文末「数字怎么来的」。
> 与上一版的三处实质变更：Mind2Web 改为直接使用官方多模态发行版（不再走转换流程）、
> 训练数据量改为按会话切分后的实际值、补上验证集环节。

## 一、四个数据集的定位

它们不是同类东西，分成两张表。上一版把四个并列，会让人以为可以互相替换。

### 数据源 —— 有截图、有动作标签，能进 LoRA

| 数据集 | 定位 | 用于 LoRA | 状态 |
|---|---|---|---|
| **ScreenAgent** | 核心训练数据源 + 同分布测试 | ✅ 是 | 已就绪，本机 4012 条 |
| **Mind2Web** | 阶段 2 增量训练数据 | ⏳ 待定 | **需先做视口裁剪**，见 §3 |

### 量具 —— 不进训练，用来测量或提供方法

| 数据集 | 定位 | 用于 LoRA | 状态 |
|---|---|---|---|
| **ScreenSpot** | 零样本 grounding 基准（分布外） | ❌ 否 | 已下载，未跑 |
| **WebArena** | 自动化评价方法参考 | ❌ 否 | 已抽样，方法已借用 |

> **为什么不把 ScreenSpot 整个删掉。** 「不用于训练」不等于「不用」。删掉会丢两样东西：
> 大纲 W3 点名的三个模型（Qwen-VL / GLM-4V / Llama 3.2 Vision）横评失去唯一的尺子；
> 以及全项目唯一的**分布外**证据 —— ScreenAgent test 虽与 train 三层隔离，
> 但仍是同一批标注者、同样 1024×768、同样的任务家族。

---

## 二、ScreenAgent：核心训练数据源

**用词**：写「核心训练数据源」，不写「唯一训练来源」—— 阶段 2 会加入 Mind2Web，
「唯一」会与后文冲突。

### 2.1 放宽 action 过滤

现有逻辑（`data/split.py:81`）：

```python
and s.resolve_point() is not None
```

问题：只有带坐标的样本能通过，`type` / `key` / `done` / `wait` / `scroll` 全部丢失。

根因是这个函数写于「grounding 层」时期，那时坐标就是任务定义本身；2026-08-24
目标改为动作生成后，过滤条件没跟着改。

改为按动作类型校验各自的必需字段：

| Action | 必需字段 |
|---|---|
| click / double_click / move | 坐标 |
| scroll | 方向（scroll_repeat） |
| type | 文本（keyboard_text） |
| key | 键名（keyboard_key） |
| wait | 秒数（wait_time） |
| done | 无 |
| drag | **仍然丢弃** —— 原始标注只有起点没有终点，构不成可执行的拖拽 |

### 2.2 统一 schema 要加参数位（上一版遗漏，最上游的一处）

`UnifiedSample` 目前只有 `bbox` / `point` 两个参数位，因为它同样是按 grounding 设计的。
装载时 `keyboard_text`、`keyboard_key`、`wait_time`、`situation` 全部丢失 ——
而原始文件里都在：

```json
{"action_type":"KeyboardAction","keyboard_action_type":"text","keyboard_text":"Hello, world!"}
{"action_type":"KeyboardAction","keyboard_action_type":"press","keyboard_key":"Ctrl+A"}
{"action_type":"WaitAction","wait_time":1.0}
```

**不改这一处，下游放宽了也只能训出「知道要打字但不知道打什么」的模型。**

### 2.3 修改指令来源

不要只用 `task_prompt`（会话级总目标）。改为当前子任务，让模型学的是：

> 当前屏幕 + 当前子任务 → 下一步动作

而不是「总任务 → 固定流程」。这同时解决旧问题：685 条样本只对应 169 个不重复指令。

取法（实测覆盖率）：

- **test 划分**：639 个文件 100% 带 `current_task` 字段，直接取
- **train 划分**：475 个带该字段，其余从 `send_prompt` 正则解析，合计覆盖 **90%**（1793 / 2000）

### 2.4 数据量

**报告中不要写「从 685 增加到 2158」** —— 2158 是放宽后的**动作池**，不是训练集。
按会话 8:2 切分后的实际值：

| | 会话 | 执行器 | 规划器 | 合计 |
|---|---:|---:|---:|---:|
| 训练 | 162 | **1705** | 707 | **2412** |
| 验证 | 41 | 431 | 177 | 608 |
| 测试（ScreenAgent 官方 test） | 70 | 646 | 284 | 930 |

对照：现有 `train.jsonl` 543 + `val.jsonl` 142 = 685。**执行器训练样本 3.1 倍。**

放宽后的动作池分布（train 划分，203 个会话）：

| 动作 | 条数 | 现状 |
|---|---:|---|
| done（子任务完成判定） | 875 | 新增 |
| click | 520 | 已有 |
| key | 254 | 新增 |
| **type** | **216** | **新增 —— 端到端瓶颈所在** |
| double_click | 91 | 已有 |
| mouse_move | 74 | 已有 |
| wait | 65 | 新增 |
| scroll | 41 | 新增 |
| plan（任务拆解） | 884 | 新增，供规划器 |

丢弃：drag 25、down/up 6、缺截图的标注文件 165。

英文表述按 GUI Agent 领域惯例以 action step 计：

> After preprocessing, 1,705 action-level training samples were obtained from 162 sessions.

---

## 三、Mind2Web：直接使用官方多模态发行版

**上一版写的「通过 ScreenAgent 的转换流程 + Globus raw_dump」是不必要的，整节作废。**

官方已发布多模态版本 **`osunlp/Multimodal-Mind2Web`**，字段中直接包含所需的一切：

```
screenshot                                      Image
pos_candidates[].attributes.bounding_box_rect   像素框
operation                                       {"op":"CLICK"|"TYPE"|"SELECT","value":...}
confirmed_task                                  任务描述
```

规模：

| split | 行数 | 大小 |
|---|---:|---:|
| train | 7775 | 7.58 GB |
| test_domain | 4060 | 3.62 GB |
| test_task | 1339 | 1.28 GB |
| test_website | 1019 | 1.10 GB |

### 3.1 但它不能直接混训 —— 画幅问题

它是**整页长截图**（网页从头滚到尾），`bounding_box_rect` 也在整页坐标系里。

跨全表取样实测（6 个 offset × 3 行，覆盖 9 个不同任务、6 个网站）：

```
宽度      1280（18 例中 1 例为 1410）—— 基本固定
高度      2660 ~ 17381              —— 相差 6.5 倍
高宽比    2.08 ~ 13.58   中位数 4.24
```

> **取样口径注意。** 用 `first-rows` 接口拿开头几行会全部落在同一个任务上
> —— Mind2Web 一行是一个动作步，同任务的连续步共用同一张页面截图，
> 于是「6 张」实际只是 1 个页面。必须跨 offset 取样。

关键不在于"高"，而在于**画幅本身不稳定**：

| | 宽高比 | 一致性 |
|---|---|---|
| ScreenAgent 训练数据 1024×768 | 0.75 | **4012/4012 完全一致** |
| 虚拟机客机 1920×1080（推理时） | 0.56 | 固定 |
| **Mind2Web** | **2.08 ~ 13.58** | **同一网站内都能差 4 倍** |

M3 §9 的分辨率消融已量出：坐标误差从 118.4 涨到 423.3，仅因分辨率偏离训练值。
再叠加 `DEFAULT_MAX_PIXELS = 896×896`：最极端的 1280×17381（2224 万像素）
需乘 0.19，变成 243×3299 —— 一个网页按钮不到 1 像素高。

**因此视口裁剪不只是"对齐画幅"，是"给它一个画幅"** —— 它现在没有固定画幅。
裁成 1280×720 视口块，bbox 落在视口外的样本丢弃。裁完宽高比 0.56，正好与客机一致。

### 3.2 它补的是量，不是种类

Mind2Web 的动作只有 CLICK / TYPE / SELECT —— 没有 done、wait、double_click、scroll，
SELECT 在桌面上也没有对应动作。而当前缺的正是动作**种类**，ScreenAgent 自己就能补上。

**结论：排在阶段 2，作为独立消融档（ScreenAgent vs ScreenAgent + Mind2Web），
不与阶段 1 混做。**

报告表述：

> Mind2Web screenshots are full-page captures with a fixed width of 1280 px but highly
> variable height (2,660–17,381 px; aspect ratio 2.08–13.58). They were viewport-cropped
> to 1280×720 before being merged into training, and actions whose target bounding box
> fell outside the cropped viewport were discarded.

---

## 四、WebArena：评价方法参考

不作为 LoRA 训练数据。它提供的是 task / environment / evaluation function，
而不是 image / action label —— **没有截图字段，也不存在另行分发的截图包**，
它是给活的 Docker 沙箱用的任务定义，设计上就不产出图。

正确用途：

- 成功率计算口径
- 自动状态验证（其 `eval` 字段：查 URL、查页面文本、调站点 API 核对状态）
- 任务完成判定

本项目的程序化成功判定方法即借鉴自此，记录在 `docs/网页数据集抽样查看.md`。
大纲 W7 要求「设计 20 个不同难度的桌面任务」，其 812 条任务模板是设计参考。

---

## 五、ScreenSpot：冻结，不参与训练

定位：**零样本 grounding benchmark（分布外）**。

不要：ScreenSpot → LoRA 训练。否则构成测试集污染，相关结论全部作废。
这既是本项目 M0 全局约束 2，也是数据集作者的明文规定：

> This dataset is a benchmarking dataset. **It is not used for training.**

补充事实，供报告说明用：

- ScreenSpot **不在官方大纲的数据集列表内**，是本项目自行加入的一条已登记偏差（`milestone-0:255`）
- 来源：南京大学 + 上海 AI Lab，SeeClick 论文（arXiv 2401.10935），Apache-2.0，需保留声明
- v1 与 v2 之间截图重叠 **586 张（占 v1 的 96.1%）**，两者不可当作两个独立评测集报告平均分
- 与 ScreenAgent 的截图重叠为 **0**

用途：给大纲 W3 点名的三个模型（Qwen-VL / GLM-4V / Llama 3.2 Vision）做零样本定位横评。

---

## 六、最终实验流程

```
阶段 1 —— 不依赖任何下载，可立即执行
────────────────────────────────────────────
  ScreenAgent（放宽过滤 + 补参数位 + 改指令来源）
        |
        +---- 训练 1705 执行器 / 707 规划器  ──> LoRA 微调
        |
        +---- 验证  431 / 177       ──> 调参、选提示词（唯一允许碰的集合）
        |
        +---- 测试  646 / 284       ──> 配置定死后跑一次
        ↓
  评测：ScreenAgent test（同分布） + 端到端五任务（虚拟机）


阶段 2 —— 可选增量，需先下载 7.58 GB 并完成视口裁剪
────────────────────────────────────────────
  Multimodal-Mind2Web ──裁剪 1280×720──> 与阶段 1 数据合并
        ↓
  独立消融档：ScreenAgent  vs  ScreenAgent + Mind2Web


测量（全程不进训练）
────────────────────────────────────────────
  1. ScreenAgent test        ──> 任务执行能力（同分布）
  2. ScreenSpot              ──> UI grounding 能力（分布外）
  3. 自建 20 个 Desktop Task ──> 真实系统能力
  4. WebArena                ──> 评价方法参考（不产出分数）
```

> **验证集与测试集的分工必须守死。** 调参、选提示词、看曲线只能用验证集；
> 测试集在配置最终确定后跑一次。当前 M3 报告中的「微调前 vs 微调后」数字跑的是
> `val.jsonl`（142 条验证集），改完后需在 ScreenAgent test 上重跑，**两个数都报**，
> 差值本身就说明选型偏高了多少。

---

## 七、报告推荐表述

> We use ScreenAgent as the primary training dataset because it provides screen
> observations and executable GUI actions. Its official train/test split is isolated at
> the session, task, and screenshot level (zero overlap on all three), and is therefore
> used as our in-distribution test set. Mind2Web is available in a multimodal release
> with full-page screenshots and pixel-level bounding boxes; it is viewport-cropped
> before being merged as a second-stage augmentation. WebArena is not used for training
> because it is designed as an interactive evaluation environment rather than a
> supervised dataset. ScreenSpot is reserved as a zero-shot grounding benchmark to
> evaluate out-of-distribution generalization.

---

## 八、实验前确认事项

### 8.1 ScreenAgent train/test 是否真正 session-level 隔离 —— ✅ 已验证

实测结果比要求的更强，三层交集全为 0：

```
会话目录     train 203    test 70    交集 0
任务描述     train 179    test 70    交集 0      （不重复任务数）
截图 MD5     train 1948   test 624   交集 0
```

### 8.2 Mind2Web 转换质量检查清单（阶段 2 执行前）

- [ ] click 坐标是否准确（`bounding_box_rect` → 中心点）
- [ ] TYPE 是否保留 `operation.value` 里的文本
- [ ] SELECT 如何处理（桌面无对应动作，建议丢弃）
- [ ] **bbox 是否落在裁剪后的视口内** —— 落在视口外的必须丢弃
- [ ] **画幅一致性** —— 裁剪后是否统一为 1280×720；与 ScreenAgent 的 1024×768
      混训时，坐标归一化的分母必须各自正确

### 8.3 阶段 1 的代码改动清单

| # | 位置 | 改什么 |
|---|---|---|
| ① | `data/schema.py` + `data/loaders/screenagent.py` | 加 `params` 字段承载动作参数，重新生成 `unified.jsonl` |
| ② | `data/split.py:81` | 按动作类型验必需字段，不再只认坐标；函数改名（已不是 grounding） |
| ③ | `finetune/dataset.py:146` | 第二道坐标门 + 扩输出格式覆盖 type / key / wait / done |
| ④ | `finetune/dataset.py:133` | 指令来源改为当前子任务 |

**执行顺序：先做 ①，重新生成 `unified.jsonl` 后核对分布**（确认 216 条 type
的文本都带上了），再往下走。这一步错了，后面全是白训。

**重训时守单变量**：数据换了就不要同时动 `r` / `alpha` / 学习率。LoRA 调参是另一次实验。

---

## 数字怎么来的

| 数字 | 来源 |
|---|---|
| 4012 / 3073 / 939 | `data/processed/unified.jsonl` 按 `source_dataset` 与 `split` 计数 |
| 各动作条数、1705 / 431 / 646 | 按 §2.1 规则在 `data/raw/screenagent_repo/data/ScreenAgent` 原始标注上实跑，会话级 8:2 切分（种子 20260913） |
| 三层交集 0 | 会话目录名、`task_prompt_en`、截图文件 MD5 三路取交集 |
| 90% 子任务覆盖率 | `current_task` 字段优先，缺失时正则解析 `send_prompt` |
| ScreenAgent 全为 1024×768 | 两条独立验证：`unified.jsonl` 的 `resolution` 字段 4012/4012；PIL 直接读随机 300 张图片文件（种子 7）300/300。**两条要分开验**——声明值与真实值不一致时坐标会静默错位，不报错 |
| Multimodal-Mind2Web 行数与体积 | HuggingFace datasets-server `/size` 接口 |
| Mind2Web 高度 2660~17381 / 高宽比 2.08~13.58 | datasets-server `/rows` 接口，offset 取 0/1500/3000/4500/6000/7000 各 3 行，逐张下载后用 PIL 读尺寸，覆盖 9 个任务、6 个网站。**不能用 `/first-rows`**——开头几行属同一任务、共用同一张截图 |
| ScreenSpot 重叠 586 / 0 | M2 已完成的泄漏检查，记录于 `milestone-2:123` |
