"""截图引擎诊断 —— 逐个试，看谁能用、有多快、画面对不对。

    python scripts/diagnose_capture.py

## 为什么需要它

`ScreenCapturer` 有三级降级：dxcam → mss → pyautogui。平时这是好事——
某台机器上 DXGI 不可用，系统照样跑得起来。但**降级是静默的**，你只会看到
"有点慢"，不会看到"为什么慢"。

客机首测就撞上了这个：实测 p50 500ms，而宿主机 dxcam 是 4ms、预期的 mss
降级也只有 24-37ms。500ms 说明降到了最后一级，或者虚拟显卡本身有问题——
这两种情况的处理方式完全不同，得先分清。

## 顺带查画面是不是黑的

虚拟机里关掉 3D 加速后，某些配置下截图会**全黑**——不报错，就是一张纯黑
图片。这种故障最阴险：Agent 会对着一张黑图问模型"下一步点哪"，模型胡乱
给个坐标，然后点在真实界面的随机位置上。

所以本脚本除了测速，还会统计画面的非零像素占比与均值。全黑时直接喊出来。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "src")
)  # 交付包把两个包放在 src/ 下

ENGINES = ("dxcam", "mss", "pyautogui")
ROUNDS = 5


def _console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def probe(name: str) -> None:
    from perception.capture import ScreenCapturer

    try:
        capturer = ScreenCapturer(prefer=name)
    except Exception as exc:  # noqa: BLE001
        print(f"  {name:<10} 创建失败: {type(exc).__name__}: {str(exc)[:90]}")
        return

    try:
        shot = capturer.capture(fresh=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  {name:<10} 截图失败: {type(exc).__name__}: {str(exc)[:90]}")
        return

    # prefer 只是把它排到最前，用不了仍会降级。这里点破实际用的是谁，
    # 否则会误以为"dxcam 能用，只是要 500ms"。
    actual = shot.engine
    note = f"（**降级了**，实际是 {actual}）" if actual != name else ""

    times = []
    for _ in range(ROUNDS):
        start = time.perf_counter()
        capturer.capture(fresh=True)
        times.append((time.perf_counter() - start) * 1000.0)
    times.sort()

    image = shot.image
    nonzero = float((image > 8).mean()) * 100.0
    mean = float(image.mean())

    print(
        f"  {name:<10} -> {actual:<10} {shot.width}x{shot.height}  "
        f"中位 {times[len(times) // 2]:7.1f}ms  最快 {times[0]:7.1f}ms  {note}"
    )
    print(f"  {'':<10}    画面：非零像素 {nonzero:.1f}%，均值 {mean:.1f}", end="")
    if nonzero < 1.0:
        print("   ★★ 几乎全黑，这个引擎的画面不可用 ★★")
    elif nonzero < 10.0:
        print("   ★ 画面偏暗，确认一下是不是正常桌面")
    else:
        print("   看起来正常")


def main() -> int:
    _console()
    from perception.dpi import describe as dpi_describe
    from perception.dpi import enable_dpi_awareness

    # **必须在读 DPI 之前启用感知。** 未声明感知时 Windows 会对进程报回
    # 逻辑值——150% 的机器会报成 96 DPI / 100%。M1 的 verify_control.py
    # 当初就栽在这，把一台 150% 的机器记成了 100%，三档 DPI 的记录全废。
    # `ScreenCapturer` 构造时会启用，但那发生在下面，这里得自己先调。
    enable_dpi_awareness()

    print("=" * 72)
    print("截图引擎诊断")
    print("=" * 72)
    dpi = dpi_describe()
    print(
        f"DPI {dpi.get('system_dpi')}（缩放 {dpi.get('scale_factor'):.0%}，"
        f"感知={dpi.get('dpi_aware')}）\n"
    )

    for name in ENGINES:
        probe(name)
        print()

    print("=" * 72)
    print("怎么读这个结果")
    print("=" * 72)
    print("  · 宿主机 dxcam 实测 p50 约 4ms，mss 约 24-37ms，pyautogui 最慢")
    print("  · 客机里 dxcam 用不了是预期的（DXGI 依赖显卡驱动）")
    print("  · 但如果 mss 也到了几百毫秒，那不是正常降级，是虚拟显卡配置问题")
    print("    —— 试试 VMware 设置里打开「加速 3D 图形」再跑一次")
    print("  · 任何一行报「几乎全黑」，那个引擎就不能用，哪怕它很快")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
