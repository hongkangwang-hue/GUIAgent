# M1 控制层真机验证记录

> 由 `python scripts/verify_control.py` 生成，每跑一次追加一节。
> 对应 M1 验收标准 3（三档 DPI）、5（动作与中文输入）、
> 6（宿主机→虚拟机通路）、7（两道急停）。

## 阅读说明

本文件由脚本逐次追加，**包含调试过程中的失败记录**。前三条不是系统缺陷，
是验证脚本自身的缺陷，均已修复：

| 记录 | 表面现象 | 真实原因 | 修复 |
|---|---|---|---|
| 22:33 | DPI 记成 100%（实为 150%） | `dpi_describe()` 在 `enable_dpi_awareness()` 之前调用，未启用感知时 Windows 会对进程报回逻辑值 | commit `ecfbbfb` |
| 22:36 | 12 点中 11 点 FailSafeException | 坐标链检查故意测四个角，而屏幕四角正是 PyAutoGUI FAILSAFE 的触发点——两条验收要求（3 与 7）在此冲突 | commit `14d376a`：角点改用纯计算验证，真移鼠标的点内缩 1 格 |
| 22:40 | 孤立一点偏 7px | 读数时物理鼠标被碰到。系统性坐标错误会让 12 点全偏且成比例，孤立一点偏移不符合该形态 | commit `28cd267`：连读两次检测外力干扰，不稳则重试 |

**22:42 那条是修复后的有效结果**：12 个目标点最大偏差 0.0px，无干扰重试。

保留失败记录而不删除，是因为它们本身说明了一件事——这三个缺陷都只有
真机运行才暴露得出来，单元测试全绿也照样存在。

---

## 2026-08-22 22:33 —— 宿主机 150% DPI

- DPI 缩放：**100%**（96 DPI，感知=False）
- 截图区域：(0, 0, 2560, 1600)
- 结果：**1/1 项通过**

| 检查项 | 结果 | 详情 |
|---|---|---|
| 急停热键 | 通过 | 触发到动作被拒绝 247ms（Agent 正控制鼠标时） |

---

## 2026-08-22 22:36 —— 宿主机 150% DPI

- DPI 缩放：**150%**（144 DPI，感知=True）
- 截图区域：(0, 0, 2560, 1600)
- 结果：**0/1 项通过**

| 检查项 | 结果 | 详情 |
|---|---|---|
| 坐标链端到端精度 | **未通过** | 12 个目标点，最大偏差 0.0px（容差 2px）；(0, 384) 执行失败：PyAutoGUI fail-safe triggered from mouse moving to a corner of the screen. To disable this fail-safe, set pyautogui.FAILSAFE to False. DISABLING FAIL-SAFE IS NOT RECOMMENDED.；(0, 767) 执行失败：PyAutoGUI fail-safe triggered from mouse moving to a corner of the screen. To disable this fail-safe, set pyautogui.FAILSAFE to False. DISABLING FAIL-SAFE IS NOT RECOMMENDED.；(256, 0) 执行失败：PyAutoGUI fail-safe triggered from mouse moving to a corner of the screen. To disable this fail-safe, set pyautogui.FAILSAFE to False. DISABLING FAIL-SAFE IS NOT RECOMMENDED. |

---

## 2026-08-22 22:40 —— 宿主机 150% DPI

- DPI 缩放：**150%**（144 DPI，感知=True）
- 截图区域：(0, 0, 2560, 1600)
- 结果：**0/1 项通过**

| 检查项 | 结果 | 详情 |
|---|---|---|
| 坐标链端到端精度 | **未通过** | 12 个目标点，最大偏差 7px（容差 2px）；模型(1, 384) 期望屏幕(2, 800) 实际(7,807) 偏差7px |

---

## 2026-08-22 22:42 —— 宿主机 150% DPI

- DPI 缩放：**150%**（144 DPI，感知=True）
- 截图区域：(0, 0, 2560, 1600)
- 结果：**1/1 项通过**

| 检查项 | 结果 | 详情 |
|---|---|---|
| 坐标链端到端精度 | 通过 | 12 个目标点，最大偏差 0.0px（容差 2px） |
