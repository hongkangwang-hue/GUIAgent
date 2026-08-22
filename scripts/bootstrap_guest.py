"""客机一键引导 —— 装依赖、冻结版本、记录环境。

明天在客机里跑这一条就够了：

    python scripts/bootstrap_guest.py

它做四件事：

1. 装 M1 验收所需的最小依赖集（不装 torch / paddleocr / langchain）
2. `pip freeze` 冻结版本到 `requirements-vm.lock.txt`
3. 调 `env_check.py` 做自检
4. **把客机的环境实测值写成验收材料** —— `docs/m1-guest-environment.md`

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

#: M1 验收所需的最小依赖集，与 CI 一致（`.github/workflows/ci.yml`）。
#: **不含 torch / paddleocr / easyocr / langchain** —— 那几个加起来好几个 GB，
#: M1 验收只用得到控制层，装了只会让快照变大。
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


def install() -> bool:
    print("=" * 64)
    print("1/4  安装依赖")
    print("=" * 64)
    pip = [sys.executable, "-m", "pip", "install", "--quiet"]
    if _run([*pip, "--upgrade", "pip"]).returncode != 0:
        return False
    for group in (RUNTIME_DEPS, DEV_DEPS):
        if _run([*pip, *group]).returncode != 0:
            print("\n[失败] 依赖安装未完成。检查网络后重跑本脚本。")
            return False
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
    """实测截图引擎、延迟、分辨率与 DPI。"""
    print("=" * 64)
    print("4/4  环境实测")
    print("=" * 64)

    from perception.capture import ScreenCapturer
    from perception.dpi import describe as dpi_describe
    from perception.dpi import enable_dpi_awareness

    # 必须在读 DPI 之前启用感知，否则 Windows 会对进程报回逻辑值。
    # M1 曾因此把 150% 的机器记成 100%。
    enable_dpi_awareness()
    dpi = dpi_describe()

    capturer = ScreenCapturer()
    shot = capturer.capture(fresh=True)

    import time

    samples = []
    for _ in range(30):
        start = time.perf_counter()
        capturer.capture(fresh=True)
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()

    data = {
        "engine": shot.engine,
        "resolution": f"{shot.width}×{shot.height}",
        "dpi": dpi.get("system_dpi"),
        "scale": dpi.get("scale_factor"),
        "dpi_aware": dpi.get("dpi_aware"),
        "p50_ms": samples[len(samples) // 2],
        "p95_ms": samples[int(len(samples) * 0.95)],
        "max_ms": samples[-1],
    }

    print(f"  截图引擎   {data['engine']}")
    print(f"  分辨率     {data['resolution']}")
    print(f"  DPI        {data['dpi']}（缩放 {data['scale']:.0%}，感知={data['dpi_aware']}）")
    print(
        f"  截图延迟   p50 {data['p50_ms']:.2f}ms  "
        f"p95 {data['p95_ms']:.2f}ms  max {data['max_ms']:.2f}ms"
    )
    if data["engine"] != "dxcam":
        print("\n  [注意] 未使用 dxcam。虚拟机里这是预期的（DXGI 依赖显卡驱动），")
        print("         已自动降级。这个延迟与宿主机的 dxcam 数据要并列呈现，")
        print("         不是缺陷，但也不能拿它替换宿主机的结论。")
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
        f"| 截图延迟 p50 | **{env['p50_ms']:.2f} ms** |",
        f"| 截图延迟 p95 | {env['p95_ms']:.2f} ms |",
        f"| 截图延迟 max | {env['max_ms']:.2f} ms |",
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
        f"| 客机 | {env['engine']} | {env['resolution']} | {env['p50_ms']:.2f} ms |",
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
    args = parser.parse_args()

    if not args.skip_install and not install():
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
    print("  1. 客机关机 → 拍快照 eval-baseline")
    print("  2. 开机，打开一个记事本")
    print("  3. python scripts\\verify_control.py --only stop")
    print("     按 Ctrl+Alt+Q，看有没有反应")
    print("     —— 没反应就是被 VMware 的 Ctrl+Alt 前缀截走了，告诉我，我换热键")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
