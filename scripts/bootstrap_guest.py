"""客机一键引导 —— 装依赖、冻结版本、记录环境。

明天在客机里跑这一条就够了：

    python scripts/bootstrap_guest.py

它做四件事：

1. 装 M1 验收所需的依赖（默认含 OCR，见下）
2. `pip freeze` 冻结版本到 `requirements-vm.lock.txt`
3. 调 `env_check.py` 做自检
4. **把客机的环境实测值写成验收材料** —— `docs/m1-guest-environment.md`

## 为什么默认装 OCR

一开始这里写的是「只装控制层，不装 paddleocr，省得快照变大」。**那是错的。**

M1 验收标准 4 要的是「在 20 张测试截图上报告**双通道融合后**的 UI 元素
召回率，**按来源分项统计**」——UIA 一条通道，OCR 一条通道，缺一条就没有
「融合」也没有「分项」。`capture_gallery.py` 正是靠这两条通道出图的。

而且它**必须在拍 `eval-baseline` 快照之前装好**：快照之后再装，每次恢复
快照都会把它抹掉，M2-M5 每轮评测都要重装一次。

代价是快照大几个 GB。100GB 的盘装得下，比每次恢复后重装划算得多。

`--no-ocr` 可以跳过（只跑验收标准 3/5/6 的控制层验证时用），但那样
标准 4 做不了。

## 第 4 步为什么单独做

M1 验收标准 1 的「截图延迟低于 15ms」是在宿主机用 dxcam 测的（2.35–5.16ms）。
客机里 dxcam 大概率不可用，会降级到 mss，延迟会高一个数量级。

**这不是缺陷，是两个不同环境的两个数字**，但必须如实并列呈现，而不是
只报好看的那个、也不是让客机的数字把宿主机的结论盖掉。手工记容易漏、
容易记错，所以让脚本生成。

同一份记录还包含分辨率与 DPI —— M2 到 M5 的每条轨迹都会记这三样
（`TrajectoryMeta.environment`），这份文件是它们的基准对照。

## 只在客机里跑

宿主机不需要——宿主机的环境记录已经在 `docs/新机器实测数据.md` 里。
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#: 控制层依赖，与 CI 一致（`.github/workflows/ci.yml`）。
#: 支撑 M1 验收标准 3（三档 DPI）、5（键鼠动作与中文输入）、6（完整通路）。
RUNTIME_DEPS = [
    "numpy",
    "pillow",
    "opencv-python",
    "mss",
    "pyautogui",
    "pynput",
    "pyperclip",
    "uiautomation",
    "dxcam",
]
DEV_DEPS = ["pytest", "rich", "typer", "pyyaml"]

#: OCR 识别通道。**M1 验收标准 4 需要它**——那条要的是「双通道融合后的
#: 召回率，按来源分项统计」，只有 UIA 一条通道既谈不上融合也谈不上分项。
#:
#: 装 CPU 版就够：单张图从几十毫秒变几百毫秒，而召回率评测总共只跑 20 张。
#: 客机没有 GPU 直通，装 GPU 版也用不上。
OCR_DEPS = ["paddlepaddle", "paddleocr"]

#: 模型调用。M1 用不到（M1 全程没有大模型参与），M2 开始才需要。
#: 放在这里是为了让 `--with-llm` 能一次装齐，避免快照拍完才发现缺东西。
LLM_DEPS = ["langchain", "langchain-openai", "python-dotenv"]

LOCKFILE = Path("requirements-vm.lock.txt")
RECORD = Path("docs/m1-guest-environment.md")


def _console() -> None:
    """Windows 控制台默认 GBK，中文之外的符号会直接抛 UnicodeEncodeError。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


#: 子进程一律按 UTF-8 解码。
#:
#: Windows 上 `subprocess.run(text=True)` 默认用系统区域编码（简体中文机器
#: 上是 GBK），而 pip 与我们自己的脚本都输出 UTF-8——不指定就会在遇到
#: 任何非 GBK 字符时抛 UnicodeDecodeError，把整个引导流程带崩。
#: 本项目在 M1 已经因为编码问题崩过三次，这是第四种形态。
_TEXT = {"encoding": "utf-8", "errors": "replace", "text": True}


def _run(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("  $ " + " ".join(args))
    return subprocess.run(args, **{**_TEXT, **kwargs})


def install(with_ocr: bool = True, with_llm: bool = False) -> bool:
    print("=" * 64)
    print("1/4  安装依赖")
    print("=" * 64)
    pip = [sys.executable, "-m", "pip", "install", "--quiet"]
    if _run([*pip, "--upgrade", "pip"]).returncode != 0:
        return False

    groups = [("控制层", RUNTIME_DEPS), ("开发工具", DEV_DEPS)]
    if with_ocr:
        groups.append(("OCR 通道（验收标准 4 需要，装得比较久）", OCR_DEPS))
    if with_llm:
        groups.append(("模型调用（M2 才用得到）", LLM_DEPS))

    for label, group in groups:
        print(f"  -- {label}")
        if _run([*pip, *group]).returncode != 0:
            print("\n[失败] 依赖安装未完成。检查网络后重跑本脚本。")
            return False

    if not with_ocr:
        print("\n  [注意] 已跳过 OCR。M1 验收标准 4（双通道召回率）做不了，")
        print("         而且补装必须在拍 eval-baseline 快照之前，否则每次")
        print("         恢复快照都会把它抹掉。")
    print("  依赖安装完成\n")
    return True


def freeze() -> None:
    print("=" * 64)
    print("2/4  冻结版本")
    print("=" * 64)
    result = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, **_TEXT)
    LOCKFILE.write_text(result.stdout, encoding="utf-8", newline="\n")
    count = len([ln for ln in result.stdout.splitlines() if ln.strip()])
    print(f"  {count} 个包已写入 {LOCKFILE}")
    print("  之后不要 pip install --upgrade —— 软件版本本身就是实验变量\n")


def self_check() -> dict:
    print("=" * 64)
    print("3/4  环境自检")
    print("=" * 64)
    result = subprocess.run(
        [sys.executable, "scripts/env_check.py", "--json", "--skip-ocr"],
        capture_output=True,
        **_TEXT,
    )
    try:
        report = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError):
        # --json 没给出可解析输出时，把原始输出印出来供人排查，
        # 但不因此中断——环境实测才是这一步真正要拿到的东西。
        print(result.stdout[-1500:] or "(无输出)")
        print(result.stderr[-800:] or "")
        return {}

    checks = report.get("checks", [])
    required = [c for c in checks if c.get("required")]
    failed = [c for c in required if not c.get("passed")]
    warned = [c for c in checks if not c.get("required") and not c.get("passed")]

    print(f"  必过项 {len(required) - len(failed)}/{len(required)} 通过，警告 {len(warned)} 项")
    for item in failed:
        print(f"  [失败] {item.get('name')}: {str(item.get('detail', ''))[:100]}")
    for item in warned:
        print(f"  [警告] {item.get('name')}: {str(item.get('detail', ''))[:80]}")
    return report


def measure() -> dict:
    """实测截图引擎、延迟、分辨率与 DPI。

    ## 为什么报两个延迟数字

    第一版只测了 `fresh=True`，在客机上量出 p50 **500ms**，一度以为是虚拟
    显卡配置有问题。**其实是测量方法的问题。**

    dxcam 走连续捕获，`fresh=True` 会等一张**新**帧；桌面静止时没有新帧，
    它等到 `FRESH_TIMEOUT_S`（0.5 秒）超时才复用缓存。在一动不动的
    PowerShell 界面上连拍 30 张，每张都撞超时——量到的是超时时长。

    改成 `fresh=False` 又倒向另一头：直接返回缓存，宿主机上量出 0.04ms，
    等于什么都没测。

    所以两个都报，各自的含义 `ScreenCapturer.benchmark` 的文档里写得很清楚：

    - **latest（`fresh=False`）** —— "取当前画面"的耗时。桌面静止时它几乎
      只是返回缓存，宿主机上量出 0.03ms、复用率 96%——这个数字好看但没用
    - **fresh（`fresh=True`）** —— "等一帧新画面"的耗时，下限是显示器刷新
      周期，**物理上不可能更低**。**验收标准 1 对照的是它**

    为什么对照 fresh：Agent 循环里截图发生在动作执行完 + 等待界面稳定之后，
    画面刚被改过，dxcam 的后台线程早已抓到新帧，`grab()` 立刻返回——走的
    正是这条路径。而 latest 在那个时刻同样会拿到新帧，两者趋于一致；只有在
    桌面静止的**测量场景**里 latest 才会退化成读缓存。

    配套的 `reuse_rate` 是判断数字可不可信的关键：接近 1 说明这批测量几乎
    全是缓存命中，延迟数字要打折看。
    """
    print("=" * 64)
    print("4/4  环境实测")
    print("=" * 64)

    from perception.capture import ScreenCapturer
    from perception.dpi import describe as dpi_describe
    from perception.dpi import enable_dpi_awareness

    # 必须在读 DPI 之前启用感知，否则 Windows 会对进程报回逻辑值。
    # M1 曾因此把一台 150% 的机器记成 100%，三档 DPI 的记录全废。
    enable_dpi_awareness()
    dpi = dpi_describe()

    capturer = ScreenCapturer()
    shot = capturer.capture(fresh=True)

    latest = capturer.benchmark(n=50, fresh=False)
    # fresh 每次可能耗 0.5 秒超时，跑少一点
    fresh = capturer.benchmark(n=5, fresh=True)

    data = {
        "engine": shot.engine,
        "resolution": f"{shot.width}x{shot.height}",
        "dpi": dpi.get("system_dpi"),
        "scale": dpi.get("scale_factor"),
        "dpi_aware": dpi.get("dpi_aware"),
        "p50_ms": latest["p50_ms"],
        "p95_ms": latest["p95_ms"],
        "max_ms": latest["max_ms"],
        "reuse_rate": latest.get("reuse_rate"),
        "fresh_p50_ms": fresh["p50_ms"],
        "fresh_p95_ms": fresh["p95_ms"],
    }

    print(f"  截图引擎   {data['engine']}")
    print(f"  分辨率     {data['resolution']}")
    print(f"  DPI        {data['dpi']}（缩放 {data['scale']:.0%}，感知={data['dpi_aware']}）")
    print(f"  取当前画面 p50 {data['p50_ms']:.2f}ms  p95 {data['p95_ms']:.2f}ms")
    print(
        f"  等一帧新画面 p50 {data['fresh_p50_ms']:.2f}ms  "
        f"p95 {data['fresh_p95_ms']:.2f}ms   ← 验收标准 1 对照这个"
    )
    if data["reuse_rate"] is not None:
        print(f"  缓存复用率 {data['reuse_rate']:.0%}")
        if data["reuse_rate"] > 0.8:
            print("             [注意] 复用率很高，说明测量期间桌面基本静止，")
            print("             这批「取当前画面」的数字偏乐观，真实使用中会高一些。")
    print()
    return data


def write_record(env: dict, check: dict) -> None:
    checks = check.get("checks", [])
    required = [c for c in checks if c.get("required")]
    failed_names = [c.get("name") for c in required if not c.get("passed")]
    lines = [
        "# 客机环境实测记录",
        "",
        "> 由 `python scripts/bootstrap_guest.py` 在**客机内**生成。",
        "> 对应 M1 验收标准 1、5、6 的环境部分。**本文件是生成物，改脚本不改它。**",
        "",
        f"生成时间：{datetime.now():%Y-%m-%d %H:%M}",
        "",
        "## 客机环境",
        "",
        "| 项 | 实测值 |",
        "|---|---|",
        f"| 操作系统 | {platform.platform()} |",
        f"| Python | {platform.python_version()} |",
        f"| 截图引擎 | **{env['engine']}** |",
        f"| 分辨率 | **{env['resolution']}** |",
        f"| DPI | {env['dpi']}（缩放 {env['scale']:.0%}，感知={env['dpi_aware']}） |",
        f"| **等一帧新画面 p50** | **{env['fresh_p50_ms']:.2f} ms** ← 验收对照 |",
        f"| 等一帧新画面 p95 | {env['fresh_p95_ms']:.2f} ms |",
        f"| 取当前画面 p50 | {env['p50_ms']:.2f} ms（缓存复用率 "
        f"{env.get('reuse_rate') or 0:.0%}，偏乐观） |",
        "",
        "## 与宿主机的对照",
        "",
        "M1 验收标准 1 的「单帧低于 15ms」是在**宿主机**用 dxcam 测的",
        "（2560×1600，p50 2.35–5.16ms）。两组数字测的是两个不同环境，",
        "**并列呈现，不互相替换**：",
        "",
        "| 环境 | 引擎 | 分辨率 | p50 |",
        "|---|---|---|---|",
        "| 宿主机 | dxcam | 2560×1600 | 2.35–5.16 ms |",
        f"| 客机 | {env['engine']} | {env['resolution']} | {env['fresh_p50_ms']:.2f} ms |",
        "",
    ]

    if env["engine"] != "dxcam":
        lines += [
            "客机未能使用 dxcam 是**预期结果**：DXGI 桌面复制依赖显卡驱动，",
            "虚拟显卡通常不支持，`ScreenCapturer` 已自动降级。这说明降级链路",
            "本身是有效的——换一台没有 DXGI 的机器，系统照样跑得起来。",
            "",
        ]

    lines += [
        "## 两个延迟数字的区别（重要）",
        "",
        "`fresh=True` 会等一张**新**帧。桌面静止时没有新帧，dxcam 会等到",
        "`FRESH_TIMEOUT_S`（0.5 秒）超时再复用缓存。**在不动的桌面上连拍，",
        "量到的是超时时长，不是引擎速度**——本项目第一次在客机上测就栽在这，",
        "量出 p50 500ms，一度以为是虚拟显卡配置有问题。",
        "",
        "反过来 `fresh=False` 在静止桌面上直接读缓存，宿主机量出 0.03ms、",
        "复用率 96%，同样没有意义。",
        "",
        "**验收标准 1 对照「等一帧新画面」**：Agent 循环里截图发生在动作执行完",
        "且界面稳定之后，画面刚被改过，新帧就绪，走的正是这条路径。",
        "",
        "## 自检结果",
        "",
        f"- 必过项：{len(required) - len(failed_names)}/{len(required)} 通过"
        + ("" if not failed_names else f"，未通过：{'、'.join(str(n) for n in failed_names)}"),
        f"- 整体自检：{'通过' if check.get('passed') else '未通过'}",
        "",
        "详见 `python scripts/env_check.py` 的完整输出。",
        "",
        "## 依赖版本",
        "",
        f"已冻结到 `{LOCKFILE}`。**M2–M5 期间不要升级** —— ",
        "`pyautogui` / `mss` / `dxcam` / `uiautomation` 任何一个换版本都可能",
        "改变延迟或行为，而软件版本本身就是实验变量。",
        "",
    ]

    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"环境记录已写入 {RECORD}")


def main() -> int:
    _console()
    parser = argparse.ArgumentParser(description="客机一键引导")
    parser.add_argument("--skip-install", action="store_true", help="跳过依赖安装")
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="不装 OCR（M1 验收标准 4 将无法进行，且补装须在拍快照之前）",
    )
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="一并装 langchain（M2 才需要；想让快照一次到位就加上）",
    )
    args = parser.parse_args()

    if not args.skip_install and not install(with_ocr=not args.no_ocr, with_llm=args.with_llm):
        return 1
    freeze()
    check = self_check()
    try:
        env = measure()
    except Exception as exc:  # noqa: BLE001
        print(f"\n[失败] 环境实测出错：{type(exc).__name__}: {exc}")
        print("依赖可能没装全。先看上面的自检输出。")
        return 1
    write_record(env, check)

    print("\n" + "=" * 64)
    print("完成。下一步：")
    print("=" * 64)
    if args.no_ocr:
        print("  0. [先做] 补装 OCR —— 快照拍完再装的话，每次恢复都会丢")
        print("     pip install paddlepaddle paddleocr")
    print("  1. 客机关机 → 拍快照 eval-baseline")
    print("  2. 开机，打开一个记事本")
    print("  3. python scripts\\verify_control.py --only stop")
    print("     按 Ctrl+Alt+Q，看有没有反应")
    print("     —— 没反应就是被 VMware 的 Ctrl+Alt 前缀截走了，告诉我，我换热键")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
