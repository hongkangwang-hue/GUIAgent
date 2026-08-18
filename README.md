# GUI Agent — 基于多模态大模型的桌面 GUI 智能体

八周实习项目。当前进度：**M1 第 2 周（桌面感知与控制核心模块）**。

计划文档见 [milestone/](milestone/)，环境搭建见 [docs/开发环境配置文档.md](docs/开发环境配置文档.md)。

---

## 快速开始

```powershell
conda activate gui-agent
python scripts/env_check.py          # 环境自检，必过项须全绿
python -m pytest tests/ -q           # 单元测试
python scripts/capture_gallery.py --name browser --delay 5   # 生成标注截图
```

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

scripts/
├── env_check.py         环境自检（必过项 / 警告项两级）
└── capture_gallery.py   感知效果可视化图集生成
```

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
