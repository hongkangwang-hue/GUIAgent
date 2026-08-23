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
import os
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

#: 子进程的环境变量：强制它**也用 UTF-8 输出**。
#:
#: 光在父进程这边 `encoding="utf-8"` 解码是不够的——子进程若按系统区域
#: 编码（简体中文机器上是 cp936）写 stdout，父进程再按 UTF-8 去读，中文
#: 就成了一串乱码。客机首跑时 env_check 的警告文案正是这样糊掉的：
#: 「CUDA ������: torch δ��װ」。
#:
#: JSON 结构本身是 ASCII，所以解析不会失败——**只有中文细节会坏，而且
#: 不报错**。这类问题最容易一直留着没人发现。
_CHILD_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


def _run(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("  $ " + " ".join(args))
    return subprocess.run(args, env=_CHILD_ENV, **{**_TEXT, **kwargs})


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
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, env=_CHILD_ENV, **_TEXT
    )
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
        env=_CHILD_ENV,
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

    ## 口径必须和宿主机基线一致，否则两个数没法比

    本项目验收标准 1（单帧 <15ms）的既有口径写在 `scripts/env_check.py`：
    `capturer.benchmark(n=30)`，**默认 `fresh=False`**，即「取当前画面」。
    `docs/新机器实测数据.md` 记的宿主机基线 p50 0.029–0.058ms 就是这个口径。

    所以这里的**主指标也用 `fresh=False`**。踩过的坑记在这，免得再走一遍：

    - 第一版用 `fresh=True`，客机量出 p50 **500ms**，以为是虚拟显卡配置有
      问题。实际上 dxcam 走连续捕获，`fresh=True` 要等一张**新**帧，桌面
      静止时没有新帧，就等满 `FRESH_TIMEOUT_S`（0.5 秒）超时——量到的是
      超时时长。
    - 之后试过在每次截图前往控制台打字符制造画面变化，数字在 4ms 到 103ms
      之间剧烈摆动。因为那测的是**画面多久变一次**，是环境属性，不是引擎速度。
    - 而且宿主机基线从来就不是用 `fresh=True` 测的，拿它跟客机的 fresh
      比属于两种口径互比，比出来的差异没有意义。

    「等一帧新画面」仍然记录，但**明确标注为环境属性、非验收指标**——
    它反映虚拟显示器的出帧率，对理解客机性能有参考价值。
    """
    print("=" * 64)
    print("4/4  环境实测")
    print("=" * 64)

    import time

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

    # fresh 单独手测，不用 benchmark()：需要把「等到了新帧」和「等满超时」
    # 分开统计。第一版用 benchmark(n=5) 报 p95，结果 5 个样本里只要有一次
    # 撞上 0.5 秒超时，p95 就等于 500ms——那不是慢帧，是那一刻画面根本没变，
    # 两件事混进一个数字里没法解读。
    timeout_ms = getattr(capturer, "_engine", None)
    timeout_ms = getattr(timeout_ms, "FRESH_TIMEOUT_S", 0.5) * 1000.0
    got_frame, timed_out = [], 0
    for index in range(20):
        # **每次截图前先改变屏幕内容。** 这不是装饰，是测量的前提：
        # 静止桌面上「等一帧新画面」量到的是画面多久变一次，不是引擎多快。
        # Agent 循环的真实形态是「动作改变了界面 → 截图」，这里用一个字符
        # 的输出模拟那个「界面刚变过」的状态。
        print("." if index % 2 else "o", end="", flush=True)
        start = time.perf_counter()
        capturer.capture(fresh=True)
        elapsed = (time.perf_counter() - start) * 1000.0
        if elapsed >= timeout_ms * 0.9:
            timed_out += 1
        else:
            got_frame.append(elapsed)
    print()
    got_frame.sort()

    data = {
        "engine": shot.engine,
        "resolution": f"{shot.width}x{shot.height}",
        "dpi": dpi.get("system_dpi"),
        "scale": dpi.get("scale_factor"),
        "dpi_aware": dpi.get("dpi_aware"),
        "p50_ms": latest["p50_ms"],
        "p95_ms": latest["p95_ms"],
        "max_ms": latest["max_ms"],
        "budget_ms": 15.0,
        "reuse_rate": latest.get("reuse_rate"),
        "fresh_p50_ms": got_frame[len(got_frame) // 2] if got_frame else float("nan"),
        "fresh_p95_ms": (got_frame[int(len(got_frame) * 0.95)] if got_frame else float("nan")),
        "fresh_n": len(got_frame),
        "fresh_timeouts": timed_out,
    }

    print(f"  截图引擎   {data['engine']}")
    print(f"  分辨率     {data['resolution']}")
    print(f"  DPI        {data['dpi']}（缩放 {data['scale']:.0%}，感知={data['dpi_aware']}）")
    verdict = "通过" if data["p95_ms"] < data["budget_ms"] else "**未通过**"
    print(
        f"  取当前画面 p50 {data['p50_ms']:.3f}ms  p95 {data['p95_ms']:.3f}ms"
        f"   ← 验收标准 1（<15ms）{verdict}"
    )
    print(
        f"  等一帧新画面 p50 {data['fresh_p50_ms']:.2f}ms  "
        f"p95 {data['fresh_p95_ms']:.2f}ms"
        f"（{data['fresh_n']}/20 次取到新帧）—— 环境属性，非验收指标"
    )
    if data["fresh_timeouts"]:
        print(
            f"             另有 {data['fresh_timeouts']}/20 次等满 "
            f"{timeout_ms:.0f}ms 超时——那一刻画面没有任何变化，不计入延迟"
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
        f"| **取当前画面 p50** | **{env['p50_ms']:.3f} ms** ← 验收对照 |",
        f"| 取当前画面 p95 | {env['p95_ms']:.3f} ms |",
        f"| 等一帧新画面 p50 | {env['fresh_p50_ms']:.1f} ms（环境属性，非验收指标） |",
        "",
        "## 与宿主机的对照",
        "",
        "同一口径（`benchmark(n=30)`，`fresh=False`）下的两个环境：",
        "",
        "| 环境 | 引擎 | 分辨率 | p50 |",
        "|---|---|---|---|",
        "| 宿主机 | dxcam | 2560×1600 | 0.029 – 0.058 ms |",
        f"| 客机 | {env['engine']} | {env['resolution']} | {env['p50_ms']:.3f} ms |",
        "",
        "宿主机数据来自 `docs/新机器实测数据.md` 第 6 节。",
        "（该文档里另有一个 2.35ms 的数字，那是**冷启动单次截图含初始化**，",
        "原文已标注「不是稳态 p50」，不可用作基线。）",
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
        "## 两个延迟数字的区别",
        "",
        "**验收标准 1 对照「取当前画面」**（`benchmark(n=30)`，`fresh=False`）。",
        "这是本项目一贯的口径——`scripts/env_check.py` 用的就是它，",
        "`docs/新机器实测数据.md` 里宿主机的 p50 0.029–0.058ms 也是它。",
        "",
        "「等一帧新画面」（`fresh=True`）测的是**等一张新帧**要多久。桌面静止时",
        "根本没有新帧，dxcam 会等满 `FRESH_TIMEOUT_S`（0.5 秒）再复用缓存，",
        "所以这个数字反映的是**虚拟显示器的出帧率**，属于环境属性，",
        "不是感知模块的性能，也不是验收指标。",
        "",
        "> 客机首测时我们一度按 `fresh=True` 量出 p50 500ms，误判为虚拟显卡",
        "> 配置有问题。记在这里，免得后来人重走一遍。",
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
