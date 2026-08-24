"""查一件具体的事：桌面双击到底能不能启动 Edge。

## 为什么单写一个

`搜索指定内容` 5 轮全灭，轨迹显示 Planner 拆的第一步是「双击桌面的
Microsoft Edge 图标」，模型点的坐标 (33,141) 经核对正好落在那个图标上，
**但 msedge.exe 从头到尾没起来**。

而 `打开浏览器` 任务 5/5 全过——它走的是**任务栏单击**。

两者只差一个动作类型。所以问题收敛成一句可以直接验的话：
**pyautogui 的 doubleClick 能不能启动一个桌面图标。**

如果不能，那是执行层的问题（双击间隔、Win+D 之后的焦点、
资源管理器对合成双击的识别），跟模型无关；M1 的动作冒烟测试没覆盖到
「双击桌面图标启动程序」这个具体组合，所以一直没暴露。

## 用法

    python scripts/diagnose_doubleclick.py                  # 默认点 (33,141)
    python scripts/diagnose_doubleclick.py --x 33 --y 141
    python scripts/diagnose_doubleclick.py --interval 0.15  # 改双击间隔

**会真的操作鼠标。** 跑之前手离开键鼠。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time


def edge_running() -> int:
    try:
        import psutil
    except ImportError:
        return -1
    return sum(
        1
        for p in psutil.process_iter(["name"])
        if (p.info.get("name") or "").lower() == "msedge.exe"
    )


def kill_edge() -> None:
    subprocess.run(
        ["taskkill", "/F", "/IM", "msedge.exe", "/T"],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    time.sleep(1.5)


def attempt(label: str, action, wait: float) -> bool:
    kill_edge()
    print(f"\n── {label}")
    print(f"   起点：msedge.exe {edge_running()} 个")
    action()
    time.sleep(wait)
    count = edge_running()
    ok = count > 0
    print(f"   {'✓ 启动成功' if ok else '✗ 未启动'}：msedge.exe {count} 个")
    return ok


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="诊断桌面双击能否启动 Edge")
    parser.add_argument("--x", type=int, default=33, help="Edge 桌面图标的屏幕 X")
    parser.add_argument("--y", type=int, default=141, help="Edge 桌面图标的屏幕 Y")
    parser.add_argument("--interval", type=float, default=0.0, help="双击两下之间的间隔（秒）")
    parser.add_argument("--wait", type=float, default=6.0, help="点完等几秒再查进程")
    args = parser.parse_args()

    import pyautogui

    pyautogui.FAILSAFE = True

    print("=" * 60)
    print("桌面双击启动诊断")
    print("=" * 60)
    print(f"  目标坐标 ({args.x}, {args.y})    双击间隔 {args.interval}s")
    print("  **会真的操作鼠标，请勿干预。**")
    print("  急停：把鼠标甩到屏幕左上角触发 FAILSAFE")

    def show_desktop() -> None:
        pyautogui.hotkey("win", "d")
        time.sleep(1.0)

    results = {}

    # A. 复现 Agent 的做法：Win+D 显示桌面 → doubleClick
    def a() -> None:
        show_desktop()
        pyautogui.doubleClick(args.x, args.y, interval=args.interval)

    results["Win+D 后 doubleClick"] = attempt(
        "A. Win+D → doubleClick（复现 Agent 的做法）", a, args.wait
    )

    # B. 先单击选中再双击。资源管理器对「首次点击」有时只做焦点转移
    def b() -> None:
        show_desktop()
        pyautogui.click(args.x, args.y)
        time.sleep(0.4)
        pyautogui.doubleClick(args.x, args.y, interval=args.interval)

    results["先单击再 doubleClick"] = attempt("B. Win+D → 单击 → doubleClick", b, args.wait)

    # C. 选中后按回车。若 A/B 都失败而 C 成功，问题在双击的合成方式
    def c() -> None:
        show_desktop()
        pyautogui.click(args.x, args.y)
        time.sleep(0.4)
        pyautogui.press("enter")

    results["单击选中 + 回车"] = attempt("C. Win+D → 单击 → 回车", c, args.wait)

    print("\n" + "=" * 60)
    for name, ok in results.items():
        print(f"  {'✓' if ok else '✗'}  {name}")
    print("=" * 60)

    if not results["Win+D 后 doubleClick"] and any(results.values()):
        print("\n  **结论：doubleClick 本身不奏效，但别的路径可以。**")
        print("  这是执行层的问题，不是模型的问题——Planner 规划「双击桌面图标」")
        print("  是完全合理的，只是我们的执行器发不出资源管理器认得的双击。")
    elif not any(results.values()):
        print("\n  三条路径全失败。问题不在双击，在坐标或桌面焦点——")
        print(f"  先手动确认 ({args.x}, {args.y}) 上确实是 Edge 图标。")
    else:
        print("\n  doubleClick 可用。那 5 轮全灭另有原因，回去看轨迹的第 2 步。")

    kill_edge()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
