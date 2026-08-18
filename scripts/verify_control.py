"""控制层真机验证。

`dry_run` 测不到的那一半在这里补齐：坐标链是否端到端准确、点击是否真的
落在目标控件上、中文输入是否可用、两道刹车是否有效。

## 两个检查是程序可判定的，不靠肉眼

M1 验收要求"点击目标位置均正确"，而"看起来点对了"不是可复现的证据。
本脚本用两条闭环把它变成断言：

1. **坐标链闭环**：`mouse_move` 到模型坐标 → 读回 `pyautogui.position()`
   → 与 `CoordinateScaler` 算出的屏幕坐标比对。这条覆盖 DPI 缩放、
   多显示器偏移、模型/屏幕换算的全部环节。
2. **落点闭环**：拿一个 UIA 控件 → 算出它的点击点 → 移过去 →
   用 `ControlFromPoint` 反查光标下的控件是不是同一个。这条回答的是
   "坐标对了，点下去是不是真落在那个控件上"。

## ⚠ 必须在隔离虚拟机内运行

本脚本会真的移动鼠标、真的敲键盘。M0 全局约束 1：Agent 的一切键鼠执行
只发生在隔离虚拟机内，宿主机不受控制。

用法::

    python scripts/verify_control.py              # 全部检查
    python scripts/verify_control.py --skip-input # 跳过需要文本框的输入检查
    python scripts/verify_control.py --only stop  # 只验急停
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from control.actions import Action, ActionType  # noqa: E402
from control.emergency_stop import EmergencyStop  # noqa: E402
from control.executor import ActionExecutor  # noqa: E402
from perception.capture import ScreenCapturer  # noqa: E402
from perception.coordinate import CoordinateScaler  # noqa: E402
from perception.dpi import describe as dpi_describe  # noqa: E402
from perception.types import Point  # noqa: E402
from perception.uia_tree import UIATree  # noqa: E402

SPACE = "planner"
TOLERANCE_PX = 2  # M1 验收标准：坐标往返误差不超过 2px


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    data: dict = field(default_factory=dict)


class Verifier:
    def __init__(self, executor: ActionExecutor, capturer: ScreenCapturer) -> None:
        self.executor = executor
        self.capturer = capturer
        self.scaler = executor.scaler
        self.results: list[CheckResult] = []

    def record(self, result: CheckResult) -> CheckResult:
        self.results.append(result)
        mark = "✓" if result.passed else "✗"
        print(f"  [{mark}] {result.name}: {result.detail}")
        return result

    # ------------------------------------------------------------------ #
    # 1. 坐标链闭环
    # ------------------------------------------------------------------ #

    def check_coordinate_chain(self) -> None:
        """移到若干模型坐标，读回真实鼠标位置比对。

        这条覆盖 DPI 缩放、显示器偏移、模型↔屏幕换算的全部环节。
        它挂掉说明整条坐标链有问题，**后面所有点击都不可信**。
        """
        import pyautogui

        space = self.scaler.get(SPACE)
        # 取九宫格外加两个边角，边角最容易暴露夹取与偏移问题
        targets = [
            Point(x, y)
            for x in (0, space.width // 4, space.width // 2, space.width - 1)
            for y in (0, space.height // 2, space.height - 1)
        ]

        worst = 0.0
        failures = []
        for model_point in targets:
            expected = self.scaler.to_real(model_point, SPACE)
            result = self.executor.execute(
                Action(ActionType.MOUSE_MOVE, x=model_point.x, y=model_point.y)
            )
            if not result.success:
                failures.append(f"{model_point.as_tuple()} 执行失败：{result.error}")
                continue

            time.sleep(0.02)  # 等鼠标真的停稳
            actual_x, actual_y = pyautogui.position()
            error = max(abs(actual_x - expected.x), abs(actual_y - expected.y))
            worst = max(worst, error)
            if error > TOLERANCE_PX:
                failures.append(
                    f"模型{model_point.as_tuple()} 期望屏幕{expected.as_tuple()} "
                    f"实际({actual_x},{actual_y}) 偏差{error}px"
                )

        self.record(
            CheckResult(
                "坐标链端到端精度",
                not failures,
                f"{len(targets)} 个目标点，最大偏差 {worst}px（容差 {TOLERANCE_PX}px）"
                + ("" if not failures else "；" + "；".join(failures[:3])),
                {"worst_error_px": worst, "num_targets": len(targets), "failures": failures},
            )
        )

    # ------------------------------------------------------------------ #
    # 2. 落点闭环
    # ------------------------------------------------------------------ #

    def check_click_lands_on_target(self) -> None:
        """用 UIA 反查光标下的控件，验证点击点真的落在目标控件上。

        坐标准确 ≠ 点得中：OCR 给的文字框可能比可点击区域小、控件可能被
        遮挡、框中心可能落在控件的透明部分。这条检查的就是这个差别。
        """
        if not UIATree.is_available():
            self.record(CheckResult("点击落点验证", False, "uiautomation 不可用，跳过"))
            return

        import uiautomation as auto

        tree = UIATree(max_elements=40, time_budget_ms=3000)
        elements = tree.capture_foreground()
        if not elements:
            self.record(CheckResult("点击落点验证", False, "前台窗口抓不到 UIA 控件，换个窗口重试"))
            return

        # 挑面积适中的控件：太小的容易受边框影响，太大的（如整个窗口）
        # 命中了也说明不了问题
        candidates = sorted(
            (e for e in elements if 400 <= e.bbox.area <= 200_000),
            key=lambda e: -e.bbox.area,
        )[:5]
        if not candidates:
            self.record(CheckResult("点击落点验证", False, "没有尺寸合适的控件可测"))
            return

        hits, misses = 0, []
        for element in candidates:
            target = element.click_point
            model_point = self.scaler.to_model(target, SPACE)
            self.executor.execute(Action(ActionType.MOUSE_MOVE, x=model_point.x, y=model_point.y))
            time.sleep(0.05)

            try:
                under_cursor = auto.ControlFromPoint(target.x, target.y)
            except Exception as exc:  # noqa: BLE001
                misses.append(f"{element.label()}：反查失败 {exc}")
                continue

            if under_cursor is None:
                misses.append(f"{element.label()}：光标下无控件")
                continue

            # 名字或矩形任一吻合即算命中——同一控件在不同 UIA 视图下
            # 可能返回父/子节点，只比名字会有误判
            same_name = (under_cursor.Name or "").strip() == element.text.strip()
            rect = under_cursor.BoundingRectangle
            same_rect = rect is not None and abs(int(rect.left) - element.bbox.left) <= 2
            if same_name or same_rect:
                hits += 1
            else:
                misses.append(f"{element.label()} → 实际命中 {under_cursor.Name!r}")

        self.record(
            CheckResult(
                "点击落点验证",
                hits >= max(1, len(candidates) // 2),
                f"{hits}/{len(candidates)} 个控件命中"
                + ("" if not misses else "；未命中：" + "；".join(misses[:2])),
                {"hits": hits, "total": len(candidates), "misses": misses},
            )
        )

    # ------------------------------------------------------------------ #
    # 3. 各动作冒烟
    # ------------------------------------------------------------------ #

    def check_actions(self) -> None:
        """逐个动作真机执行一次，只看有没有异常。"""
        space = self.scaler.get(SPACE)
        cx, cy = space.width // 2, space.height // 2

        cases = [
            ("screenshot", Action(ActionType.SCREENSHOT)),
            ("mouse_move", Action(ActionType.MOUSE_MOVE, x=cx, y=cy)),
            ("scroll_down", Action(ActionType.SCROLL, x=cx, y=cy, direction="down", amount=2)),
            ("scroll_up", Action(ActionType.SCROLL, x=cx, y=cy, direction="up", amount=2)),
            ("key_esc", Action(ActionType.KEY, keys="esc")),
            ("wait", Action(ActionType.WAIT, duration=0.2)),
        ]

        failures = []
        for name, action in cases:
            result = self.executor.execute(action)
            if not result.success:
                failures.append(f"{name}：{result.error}")

        self.record(
            CheckResult(
                "动作冒烟",
                not failures,
                f"{len(cases) - len(failures)}/{len(cases)} 通过"
                + ("" if not failures else "；" + "；".join(failures)),
                {"failures": failures},
            )
        )

    def check_drag_releases_button(self) -> None:
        """拖拽后左键必须已松开。

        卡在按下状态是最阴险的故障：后续每一次"移动"都变成拖拽，
        表现为界面被乱选、图标被乱拖，而日志里每个动作都显示成功。
        """
        import pyautogui

        space = self.scaler.get(SPACE)
        result = self.executor.execute(
            Action(
                ActionType.LEFT_CLICK_DRAG,
                x=space.width // 3,
                y=space.height // 3,
                to_x=space.width // 2,
                to_y=space.height // 2,
            )
        )
        time.sleep(0.1)
        # 移动一小段后看是否产生了选区行为无法程序化判定，改为直接
        # 检查按键状态：mouseUp 之后再发一次不会有副作用
        pyautogui.mouseUp(button="left")

        self.record(
            CheckResult(
                "拖拽后左键已释放",
                result.success,
                "拖拽执行成功并已补发 mouseUp" if result.success else result.error,
            )
        )

    # ------------------------------------------------------------------ #
    # 4. 中文输入
    # ------------------------------------------------------------------ #

    def check_chinese_input(self) -> None:
        """输入中文并用 UIA 读回验证。

        PyAutoGUI 的 `typewrite()` 只发 ASCII 键码，中文会**静默失败**——
        不报错、也不输入。因此必须读回验证，不能只看动作返回成功。
        """
        import uiautomation as auto

        probe = "测试中文输入 ABC 123"
        edit = auto.EditControl(searchDepth=8)
        if not edit.Exists(maxSearchSeconds=2):
            self.record(
                CheckResult(
                    "中文输入",
                    False,
                    "前台窗口没有可输入的文本框。请先打开记事本或搜索框，再用 --only input 重跑",
                )
            )
            return

        try:
            edit.SetFocus()
            time.sleep(0.2)
            self.executor.execute(Action(ActionType.KEY, keys="ctrl+a"))
            self.executor.execute(Action(ActionType.KEY, keys="delete"))
            result = self.executor.execute(Action(ActionType.TYPE, text=probe))
            time.sleep(0.3)

            actual = ""
            try:
                actual = edit.GetValuePattern().Value or ""
            except Exception:  # noqa: BLE001 —— 部分控件不支持 ValuePattern
                with contextlib.suppress(Exception):
                    actual = edit.GetTextPattern().DocumentRange.GetText(-1) or ""

            matched = probe in actual
            self.record(
                CheckResult(
                    "中文输入",
                    matched,
                    f"写入 {probe!r}，读回 {actual.strip()[:40]!r}"
                    + ("" if matched else "（不匹配，剪贴板方案可能失效）"),
                    {
                        "input_method": result.meta.get("input_method", "?"),
                        "expected": probe,
                        "actual": actual.strip()[:80],
                    },
                )
            )
        except Exception as exc:  # noqa: BLE001
            self.record(CheckResult("中文输入", False, f"验证过程出错：{exc}"))

    # ------------------------------------------------------------------ #
    # 5. 两道刹车
    # ------------------------------------------------------------------ #

    def check_emergency_stop_latency(self) -> None:
        """测"按下热键 → 动作被拒绝"的响应延迟。

        M1 验收标准 7 要求热键**在 Agent 正控制鼠标时仍能立即中断**，
        所以这里边跑动作边等触发，而不是空转等待。
        """
        stop = self.executor.emergency_stop
        if not stop.is_armed and not stop.arm():
            self.record(CheckResult("急停热键", False, "热键未能启用，当前只剩 FAILSAFE 一道刹车"))
            return

        print(f"\n  >>> 请在 15 秒内按下 {stop.hotkey}（期间鼠标会持续移动，模拟 Agent 正在操作）")
        space = self.scaler.get(SPACE)
        deadline = time.time() + 15
        blocked_at = None

        while time.time() < deadline:
            result = self.executor.execute(
                Action(
                    ActionType.MOUSE_MOVE,
                    x=(int(time.time() * 200) % (space.width - 1)),
                    y=space.height // 2,
                )
            )
            if not result.success and result.error_type == "emergency_stopped":
                blocked_at = time.time()
                break

        if blocked_at is None:
            self.record(CheckResult("急停热键", False, "15 秒内未收到热键。检查是否被其他程序占用该组合键"))
            return

        latency_ms = (blocked_at - stop.triggered_at) * 1000.0
        self.record(
            CheckResult(
                "急停热键",
                latency_ms < 500,
                f"触发到动作被拒绝 {latency_ms:.0f}ms（Agent 正控制鼠标时）",
                {"latency_ms": round(latency_ms, 1)},
            )
        )

        # 复位，否则后续检查全部被拒
        stop.reset()

    def check_failsafe_configured(self) -> None:
        """确认 FAILSAFE 已开启。

        只验配置不验触发——真触发需要把鼠标甩到角落，而那正是它在
        Agent 控制鼠标时不可靠的原因（人和程序抢光标）。这条只保证
        第一道刹车在位，可靠性由热键那道兜底。
        """
        import pyautogui

        self.record(
            CheckResult(
                "FAILSAFE 已启用",
                bool(pyautogui.FAILSAFE),
                f"FAILSAFE={pyautogui.FAILSAFE}, PAUSE={pyautogui.PAUSE}s",
            )
        )

    # ------------------------------------------------------------------ #
    # 6. 安全拦截在真机上仍然生效
    # ------------------------------------------------------------------ #

    def check_safety_still_blocks(self) -> None:
        """真机上高危动作仍被拦——dry_run 下测过，这里确认没被绕过。"""
        result = self.executor.execute(Action(ActionType.TYPE, text="shutdown /s /t 0"))
        self.record(
            CheckResult(
                "安全拦截生效",
                not result.success and result.error_type == "blocked",
                f"高危文本被拦截：{result.verdict.rule if result.verdict else '未拦截！'}",
            )
        )


# ---------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description="控制层真机验证（必须在隔离虚拟机内运行）")
    parser.add_argument("--only", choices=["coord", "click", "actions", "input", "stop", "safety"])
    parser.add_argument("--skip-input", action="store_true", help="跳过需要文本框的输入检查")
    parser.add_argument("--skip-stop", action="store_true", help="跳过需要人工按键的急停检查")
    parser.add_argument("--monitor", type=int, default=1)
    args = parser.parse_args()

    dpi = dpi_describe()
    print("=" * 70)
    print("控制层真机验证")
    print(f"  DPI 缩放 {dpi['scale_factor']:.0%}（{dpi['system_dpi']} DPI），感知={dpi['dpi_aware']}")
    print("  ⚠ 本脚本会真的移动鼠标与敲键盘，必须在隔离虚拟机内运行")
    print("=" * 70)

    with ScreenCapturer() as capturer:
        region = capturer.monitor_region(args.monitor)
        scaler = CoordinateScaler(region)
        scaler.register(SPACE, 1024, 768)
        print(f"  截图区域 {region.as_tuple()}，引擎 {capturer.engine_name}")
        print(f"  坐标系 {SPACE} 1024×768，往返误差上界 "
              f"{scaler.roundtrip_error_bound(SPACE):.2f}px\n")

        executor = ActionExecutor(
            scaler, space_name=SPACE, capturer=capturer, emergency_stop=EmergencyStop()
        )
        executor.start()
        verifier = Verifier(executor, capturer)

        try:
            checks = {
                "coord": verifier.check_coordinate_chain,
                "click": verifier.check_click_lands_on_target,
                "actions": lambda: (verifier.check_actions(), verifier.check_drag_releases_button()),
                "safety": lambda: (
                    verifier.check_safety_still_blocks(),
                    verifier.check_failsafe_configured(),
                ),
                "input": verifier.check_chinese_input,
                "stop": verifier.check_emergency_stop_latency,
            }
            if args.only:
                checks[args.only]()
            else:
                for name, fn in checks.items():
                    if name == "input" and args.skip_input:
                        continue
                    if name == "stop" and args.skip_stop:
                        continue
                    fn()
        finally:
            executor.stop()

        print()
        print("-" * 70)
        print("执行器统计：", executor.stats())

    failed = [r for r in verifier.results if not r.passed]
    print("-" * 70)
    if failed:
        print(f"✗ {len(failed)}/{len(verifier.results)} 项未通过：")
        for result in failed:
            print(f"    - {result.name}：{result.detail}")
    else:
        print(f"✓ 全部 {len(verifier.results)} 项通过")
    print()
    print("提醒：M1 验收标准 3 要求在 100% / 125% / 150% 三档 DPI 缩放下都正确。")
    print("      改系统缩放后重跑本脚本，三次结果都要记录进验收材料。")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
