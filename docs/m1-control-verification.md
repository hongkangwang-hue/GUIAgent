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

## ⚠ 本文件目前全部是**宿主机**数据

M1 验收标准 5、6 要求在**隔离客机内**验证，而下面所有记录都是在宿主机上跑的。
它们作为开发阶段的记录有效，**但不能直接充当 M1 的最终验收材料**。

尤其是急停那一条（247ms）。进了虚拟机，链路多出「VMware 输入捕获」这一段，
而且 `Ctrl+Alt` 正是 VMware 释放输入焦点的默认前缀键——**这个热键有可能
根本到不了客机**，那样按下去只是把键鼠交还给宿主机，Agent 照跑不误。

客机建好后要重跑并**并列呈现两组数据**，各自注明测量环境。

此后又发生过一次同类问题：连着两次用 `--note "100% DPI"` 与 `--note "125% DPI"`
运行，脚本却都实测到 150% —— 显示缩放压根没改成功，三条记录看似三档齐了，
实际全是同一档。那两条已删除，并给脚本加了护栏：备注里的百分比与实测 DPI
不符时**拒绝写入报告**。档位标错的记录比没有记录更糟，
它会让人以为测过了。

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
---

## 2026-08-22 22:44 —— 宿主机 150% DPI

- DPI 缩放：**150%**（144 DPI，感知=True）
- 截图区域：(0, 0, 2560, 1600)
- 结果：**2/2 项通过**

| 检查项 | 结果 | 详情 |
|---|---|---|
| 安全拦截生效 | 通过 | 高危文本被拦截：dangerous_text |
| FAILSAFE 已启用 | 通过 | FAILSAFE=True, PAUSE=0.0s |
