# GUI Agent — 基于多模态大模型的桌面 GUI 智能体

八周实习项目。**M1（感知与控制）已完成，M2（端到端闭环）进行中。**

当前状态与下一步见 [docs/执行计划.md](docs/执行计划.md)，
计划文档见 [milestone/](milestone/)，
环境搭建见 [docs/开发环境配置文档.md](docs/开发环境配置文档.md)。

---

## 快速开始

### 宿主机（开发与离线实验）

```powershell
conda activate gui-agent
python scripts/env_check.py          # 环境自检，必过项须全绿
python -m pytest tests/ -q           # 单元测试
```

### 隔离客机（真机执行）

**Agent 的一切键鼠执行只发生在隔离虚拟机内。** 客机搭建见
[docs/虚拟机客机搭建说明.md](docs/虚拟机客机搭建说明.md)，装好后：

```powershell
git clone <本仓库>
cd GUIAgent
python scripts/bootstrap_guest.py    # 装依赖 + 自检 + 记录环境
python scripts/verify_control.py     # 控制层真机验证（会真的动鼠标）
```

### 跑一个任务

```powershell
copy .env.example .env               # 填入 API Key
python -m cli run "打开记事本"        # 演练，不动鼠标
python -m cli run "打开记事本" --execute   # 实机执行（二次确认 + 急停提示）
python -m cli replay                 # 回放最近一条轨迹
```

**`run` 默认是演练模式**，不加 `--execute` 不会碰键鼠。

---

## 目录结构

```
perception/          感知层
├── types.py         BBox / Point / UIElement，IoU 去重（纯标准库）
├── coordinate.py    CoordinateScaler —— 坐标转换的唯一入口（纯标准库）
├── dpi.py           Windows DPI 感知声明
├── capture.py       ScreenCapturer，dxcam / mss / pyautogui 三引擎
├── preprocess.py    OCR 预处理链（可逐步开关）
├── ocr_engine.py    OCREngine 抽象 + PaddleOCR / EasyOCR
├── uia_tree.py      Windows UIA 元素树（深度/数量/时间三重刹车）
├── element_detector.py  双通道并行 + 融合去重
└── visualizer.py    调试可视化（中文标签走 PIL）

control/             控制层
├── actions.py       动作空间 + JSON Schema（两种工具渲染方式）
├── safety.py        安全白名单（第二道防线）
├── emergency_stop.py 全局热键急停（第二道刹车）
└── executor.py      动作执行器

llm/                 模型后端（"做什么动作"）
├── base.py          LLMBackend 抽象 + 成本/用量核算
├── openai_compat.py OpenAI 兼容实现（百炼 / 智谱 / NVIDIA 共用）
├── parsing.py       容错解析（围栏、全角标点、打包坐标、bbox 折中心）
├── providers.py     多平台配置注册表
└── fake.py          ScriptedBackend，测试用

grounding/           定位后端（"点在哪里"）
├── base.py          GroundingBackend 抽象
└── native.py        NativeGrounding（模式 A：透传并校验模型坐标）

core/
├── loop.py          Agent Loop —— 截图→问模型→定位→执行→回传
└── trajectory.py    轨迹落盘 / 回放 / 失败打标

agent/               编排层
├── planner.py       指令 → 细粒度线性子任务
├── context.py       历史窗口与旧帧降采样
├── prompts.py       YAML 提示词模板库
└── session.py       串起 Planner 与 Loop

data/                数据集处理（M2 交付物 1）
├── schema.py        统一样本 schema，坐标一律绝对像素
├── loaders/         ScreenSpot / ScreenSpot-v2 / ScreenAgent
├── clean.py         清洗规则（逐条计数）
├── stats.py         统计聚合 + 跨数据集重叠检查
└── split.py         三分划定与冻结（含内容指纹）

cli/                 typer + rich
prompts/             提示词模板（executor / planner / 模式 B）

scripts/
├── env_check.py             环境自检（必过项 / 警告项两级）
├── bootstrap_guest.py       客机一键引导
├── verify_control.py        控制层真机验证
├── verify_llm.py            API 可用性验证
├── verify_mode_b.py         模式 B 可行性验证
├── capture_gallery.py       感知可视化图集
├── annotate.py              真值标注
├── eval_recall.py           召回率评测
├── ocr_benchmark.py         OCR 双引擎对照
├── make_ocr_testset.py      OCR 合成测试集
└── prepare_datasets.py      数据集下载 → 清洗 → 统计 → 冻结划分
```

---

## M1 实测结论

全部在**隔离客机**内实测（1920×1080 / dxcam），原始记录在 `docs/`：

| 项 | 结果 |
|---|---|
| 截图取帧 p50 | **0.043 ms**（宿主机同口径 0.029–0.058 ms） |
| 坐标链端到端偏差 | **0.0 px**，100% / 125% / 150% 三档 DPI 均如此 |
| 双通道元素召回率 | **100%**（5 张 / 50 个任务相关元素），UIA 72% / OCR 28% |
| 急停热键延迟 | **271 ms**（Agent 正控制鼠标时） |
| OCR 引擎选型 | PaddleOCR：真实截图快 1.55×，中/英/混排准确率均 100% |
| 单元测试覆盖率 | 85% |

三条**与直觉相反**、写进技术报告的发现：

1. **并行执行双通道反而更慢。** 客机无 GPU，PaddleOCR 占满 4 个 vCPU
   11.5 秒，把 UIA 的 COM 调用饿死——UIA 从 228ms/98 个退化到
   8133ms/61 个且被截断，分项结论从「OCR 主导」反转为「UIA 72%」。
2. **IoU 不适合评 GUI 定位。** 同一份检测结果，仅改变人画真值框的松紧，
   IoU 从 10% 变 100%，而「点击点落入真值框」判据两次都是 100%。
3. **OCR 的「中英混排准确率」是个假指标。** PaddleOCR 每个字都认对，
   只是不输出中英之间的空格；去空格归一化后从 86.7% 升到 100%。

详见 [docs/技术报告-相关工作与系统设计.md](docs/技术报告-相关工作与系统设计.md)。

---

## 三条设计上的硬规矩

**1. 坐标转换只走 `CoordinateScaler`。**
任何模块自行换算（哪怕只是一次 `x * 1024 // 1920`）都会在某个 DPI 缩放或某台显示器上错位，而且表现为"点偏了一点点"，极难定位。

**2. 隔离虚拟机是第一道防线，`safety.py` 是第二道。**
文本模式匹配可以被绕过——模型能把 `shutdown /s` 拆成两次输入。白名单拦得住"模型犯傻"，拦不住"模型有意规避"。真正的边界永远是虚拟机。

**3. 虚拟机解决执行隔离，不解决数据隐私。**
规划层走云端 API，虚拟机里的截图会被上传出去。客机按"可公开的测试环境"建：不登录真实个人账号，不放真实聊天记录、邮件、支付信息。

---

## 两处偏离原计划、有实测依据的改动

**① 增加 dxcam 作为首选截图引擎。**
M0 技术栈只列了 mss，但本机 2560×1600 实测 mss 的 `grab()` 单帧就要 41ms，**达不到 M1「单帧 <15ms」的验收标准**（1920×1080 下仍需 24ms）。改用 DXGI 桌面复制后 p50 降到 0.18ms。mss 保留为跨平台 fallback。

**② dxcam 用一次性 `grab()` + 自己缓存，而不是官方推荐的后台捕获线程。**
`get_latest_frame()` 是帧同步的，耗时被显示器刷新周期钉住（60Hz 下 16.6ms）。而 DXGI 返回 `None` 的语义是"画面没变"，此时复用缓存既最快又正确。动作发出后的观察用 `fresh=True` 强制等新帧。

| 策略 | p50 | p95 |
|---|---|---|
| mss | 37ms | — |
| dxcam `start(60)` + `get_latest_frame` | 16.63ms | 17.84ms |
| **dxcam `grab()` + 缓存** | **0.18ms** | **4.68ms** |
| dxcam 强制等新帧（`fresh=True`） | 5.96ms | 7.13ms |

---

## 测试

```powershell
python -m pytest tests/ -q --cov=perception --cov=control    # 全部
python -m pytest tests/ -q -m "not windows"                  # CI（无桌面）
python -m pytest tests/ -q -m "not slow"                     # 跳过基准测试
```

单元测试**不加载真实 OCR 模型、不移动鼠标**。前者会让本地跑测试变成需要下决心的事，后者会让跑测试变成危险操作——两种情况下最终都没人跑测试。真实模型与真实键鼠的可用性由 `scripts/env_check.py` 和虚拟机内的人工验证负责。

---

## 安全

- `.env` 已在 `.gitignore`，仓库内不得出现任何 API Key
- 急停热键默认 `Ctrl+Alt+Q`，可经 `EMERGENCY_STOP_HOTKEY` 配置
- PyAutoGUI `FAILSAFE`（鼠标甩角）与全局热键**两道刹车同时保留**，互为兜底
