# 桌面感知与控制核心模块 —— 第 2 周交付

> 交付物：**模块完整代码 + 单元测试报告**
> 生成日期：2026-08-26　　源仓库 commit：`e0504f8`

---

## 一、大纲四项任务逐条对照

| # | 大纲任务 | 实现位置 | 状态 |
|---|---|---|---|
| 1 | 实现跨平台屏幕实时截图功能，支持多分辨率适配 | `src/perception/capture.py`、`dpi.py`、`coordinate.py` | ✅ 完成（见 §3 偏差 1） |
| 2 | 集成开源 OCR 工具，实现屏幕文字与 UI 元素识别 | `src/perception/ocr_engine.py`、`preprocess.py`、`element_detector.py`、`uia_tree.py` | ✅ 完成 |
| 3 | 开发鼠标键盘控制模块，支持点击、输入、滚动、拖拽等基本操作 | `src/control/actions.py`、`executor.py`、`safety.py`、`emergency_stop.py` | ✅ 完成 |
| 4 | 实现 UI 元素坐标定位与边界框绘制功能 | `src/perception/coordinate.py`、`types.py`、`visualizer.py` | ✅ 完成 |

### 任务 1 —— 屏幕实时截图

`ScreenCapturer` 提供 **mss / dxcam 两个后端**，dxcam 在 Windows 上走
Desktop Duplication API，延迟更低；不可用时自动降级到 mss。

「多分辨率适配」由两个模块共同保证：

- `dpi.py` —— 进程启动时调用 `enable_dpi_awareness()`。**不启用的话，
  Windows 会对进程谎报逻辑分辨率**：150% 缩放下截图拿到的是 1280×800，
  而鼠标要点的是 1920×1200 的物理坐标，每一次点击都会偏。这个坑在
  `docs/m1-control-verification.md` 的 22:33 记录里有实测过程。
- `coordinate.py` —— `CoordinateScaler` 把模型输出的归一化坐标
  （\[0,1000)）映射到真实屏幕像素，支持三种缩放模式。**坐标系统是本项目
  的地基**：这一层错了，上面所有能力都失效且难以定位。

### 任务 2 —— OCR 与 UI 元素识别

`ocr_engine.py` 定义 `OCREngine` 抽象，两个实现：**PaddleOCR**（默认）与
**EasyOCR**（可切换备选）。选型不是拍的，有对照实验：

> PaddleOCR 真实截图上快 **1.55 倍**，中/英/中英混排字符准确率均 **100%**，
> 零漏检。预处理（灰度 + CLAHE）**没有收益**，两个引擎都是空跑组最好或持平。
> —— `docs/m1-ocr-双引擎对照实验小结.md`

`element_detector.py` 做的是**双通道融合**：UIA 控件树 + OCR 文本并行识别，
按 IoU 去重。两条通道互补——UIA 读得到原生控件的语义（按钮名、可交互性）
但穿不透自绘界面（浏览器内容区）；OCR 反之。

实测召回率（`docs/m1-recall-evaluation.md`，5 张截图 / 50 个元素）：

| 判据 | 召回率 |
|---|---|
| 点击点落入真值框 | **100%**（50/50） |
| IoU ≥ 0.5 | 66%（33/50） |
| 来源分布 | UIA 72% / OCR 28% |

### 任务 3 —— 鼠标键盘控制

`actions.py` 定义完整动作空间并生成 JSON Schema，大纲点名的四类全部覆盖：

| 大纲要求 | 动作 |
|---|---|
| 点击 | `left_click` / `right_click` / `double_click` / `middle_click` / `triple_click` |
| 输入 | `type`（含中文）、`key`（组合键） |
| 滚动 | `scroll` |
| 拖拽 | `left_click_drag` |

另有 `mouse_move`、`wait`、`screenshot`、`hold_key`。

`executor.py` 把结构化动作翻译成 pyautogui 调用。**中文输入不能直接
`typewrite`**——pyautogui 走的是键码模拟，中文字符没有对应键码，实现改走
剪贴板 + `ctrl+v`。

**两道独立的刹车**（超出大纲要求）：

- `safety.py` —— 执行前拦截。危险命令文本、危险组合键、桩动作。
- `emergency_stop.py` —— 全局热键急停，另有 pyautogui FAILSAFE 兜底。

### 任务 4 —— 坐标定位与边界框绘制

`types.py` 定义 `BBox` / `Point` / `UIElement` 与 `dedupe_by_iou()`；
`visualizer.py` 把识别结果画回截图（框 + 标签 + 来源着色），用于调试与
交付演示。`scripts/capture_gallery.py` 一条命令产出标注图。

---

## 二、怎么运行

### 跑单元测试（不需要真实桌面）

```bash
cd <本目录>
python -m pytest                      # 215 passed, 1 skipped
python -m pytest --cov=perception --cov=control --cov-report=term-missing
```

`pyproject.toml` 里配了 `pythonpath = ["src"]`，**不需要安装**，直接跑。

### 跑真机验证（需要 Windows 桌面）

```bash
python scripts/diagnose_capture.py                    # 截图后端与 DPI 自检
python scripts/verify_control.py --skip-input         # 坐标链 + 动作（全程不碰鼠标）
python scripts/verify_control.py --only input --delay 5   # 中文输入（倒计时里切窗口）
python scripts/capture_gallery.py                     # 识别 + 边界框绘制演示
python scripts/ocr_benchmark.py                       # OCR 双引擎对照
```

> **`verify_control.py` 会真的操作鼠标键盘。** 本项目的全局约束要求
> 一切键鼠执行只发生在隔离虚拟机内。在宿主机上跑它只用于开发自检。

### 依赖

| 包 | 用途 | 必需 |
|---|---|---|
| `mss` | 截图（跨平台后端） | 是 |
| `pyautogui` | 鼠标键盘控制 | 是 |
| `pillow` / `numpy` | 图像处理 | 是 |
| `dxcam` | 截图（Windows 低延迟后端） | 否，缺失时降级到 mss |
| `paddleocr` | OCR（默认引擎） | 否，两个 OCR 至少要有一个 |
| `easyocr` | OCR（备选引擎） | 否 |
| `pywinauto` / `comtypes` | UIA 控件树 | 否，非 Windows 自动降级为 OCR 单通道 |
| `opencv-python` | OCR 预处理 | 否 |
| `pyperclip` | 中文输入（剪贴板通道） | 中文输入需要 |
| `pynput` | 全局热键急停 | 否 |

单元测试**只需要** `pytest` / `numpy` / `pillow`，其余全部有降级路径。

---

## 三、已知偏差（必须随交付说明）

### 1. 「跨平台」的准确说法是：架构跨平台，仅在 Windows 上验证

代码里**没有 `sys.platform` 分支**，截图（mss）、鼠标键盘（pyautogui）、
OCR（PaddleOCR / EasyOCR）本身都是跨平台库。

**唯一 Windows 专有的是 UIA 识别通道**：非 Windows 上
`UIATree.is_available()` 返回 False，`ElementDetector` 自动降级为 OCR 单通道。

八周工期内没有 mac / Linux 机器做实机验证，因此**只声称「仅在 Windows 上
验证过」，不声称「支持三平台」**。

### 2. 技术栈里的 PyQt5 未采用

大纲技术栈列出 PyQt5，本项目改用 **mss + PIL** 承担截图与绘图职能。
大纲交付物只要求「简单的命令行交互界面」（第 4 周），未要求图形界面。

### 3. 召回率的 5 张样本是自设指标的下调

「召回率」在大纲第 2 周中**一次都没有出现**（大纲要的是「边界框绘制功能」
+ 「单元测试报告」）。定量测召回率是本项目自加的验收标准，20 张这个数字
也是自定的，下调到 5 张是本项目内部的取舍，不构成对大纲的偏离。

**代价要说清楚**：5 张全部是 Windows 原生应用，**未覆盖网页内容**——
那正是 UIA 读不到、只能靠 OCR 的场景。报告里的 100% 不能外推到那一类界面。

### 4. 单元测试不覆盖真实硬件路径

见 `单元测试报告.md` §3。触碰真实屏幕 / OCR 引擎 / Windows API 的代码路径
由 `scripts/` 下的真机验证脚本覆盖，其记录在 `docs/` 中。

---

## 四、目录结构

```
W2-桌面感知与控制模块/
├── README.md                    本文件
├── 单元测试报告.md               大纲要求的交付物之一
├── pyproject.toml               pytest 与覆盖率配置
├── requirements.txt             依赖清单（只跑测试的话装前四行即可）
├── src/
│   ├── perception/              感知层（11 个模块）
│   └── control/                 控制层（5 个模块）
├── tests/                       7 个测试文件，216 个用例
├── scripts/                     4 个真机验证脚本
└── docs/                        支撑实测记录
    ├── m1-ocr-双引擎对照实验小结.md      OCR 选型依据
    ├── m1-ocr-benchmark-synth.json      合成集原始数据
    ├── m1-ocr-benchmark-real.json       真实集原始数据
    ├── m1-control-verification.md       控制层真机验证记录
    └── m1-recall-evaluation.md          UI 元素召回率实测
```

`src/perception/change.py` 属于**第 6 周**交付（错误检测与自动重试），
它在 perception 包里，随包一起提供，但不计入本周的测试与覆盖率统计。
