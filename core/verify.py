"""任务成功的**程序化判定** —— M2 验收标准 2 的地基。

M2 任务 8 写死了这条：

> **成功判定必须程序化**：进程检查、窗口标题检查、文件内容检查，
> 不用人工目测。

## 为什么不能人工目测

三个理由，按严重程度排：

1. **模型自报的 `done` 不等于任务完成。** 循环收尾与任务达成是两件事，
   开源规划模型在多步任务上尤其容易提前报完成。若拿 `done` 当成功，
   成功率会被系统性高估，而且高得看不出来。
2. **人工目测不可复现。** M5 要跑 20 任务 × 3 次共 60 轮，M2 是 5 × 5，
   跨越数周。同一个人在不同时间对「算不算成功」的判断会漂移，
   而这个漂移会直接进入成功率。
3. **人工目测挡不住挂机。** 大部分评测应当无人值守地跑，
   需要有人盯着的判定形同虚设。

思路直接来自 WebArena 的功能正确性评测方法（M1 调研阶段读的）。

## 判定器只回答一个问题

「任务完成后，系统处于预期状态了吗」——不关心过程。Agent 用什么路径
达成的不重要；点了十次还是一次，只要终态对就算成功。

这也意味着**判定器必须在客机内运行**：它要查客机的进程、窗口、文件。
宿主机看不到那些，这是 M0 把 Agent 放进客机的三条理由之一。
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

IS_WINDOWS = sys.platform == "win32"


@dataclass
class CheckResult:
    """一次判定的结果。`detail` 要能让人看懂为什么判成这样。"""

    passed: bool
    detail: str
    kind: str = ""

    def __bool__(self) -> bool:
        return self.passed


# ---------------------------------------------------------------------- #
# 单项判据
# ---------------------------------------------------------------------- #


def _running(name_pattern: str) -> list[str]:
    """返回匹配的进程名列表。匹配不区分大小写、按子串。"""
    try:
        import psutil
    except ImportError:
        return []
    pattern = name_pattern.lower()
    hits = []
    for proc in psutil.process_iter(["name"]):
        try:
            name = (proc.info.get("name") or "").lower()
        except Exception:  # noqa: BLE001 —— 进程可能在遍历途中消失
            continue
        if pattern in name:
            hits.append(name)
    return hits


def check_process(name: str, should_run: bool = True) -> CheckResult:
    """进程在 / 不在。

    `should_run=False` 用于「关闭应用」这类任务——**它的判据是进程消失**，
    而不是「点到了关闭按钮」。后者用截图也能看，但点了不一定关掉。
    """
    hits = _running(name)
    ok = bool(hits) if should_run else not hits
    verb = "应在运行" if should_run else "应已退出"
    state = f"找到 {len(hits)} 个：{', '.join(sorted(set(hits))[:3])}" if hits else "未找到"
    return CheckResult(ok, f"进程 {name!r} {verb}；{state}", "process")


def check_window_title(pattern: str, regex: bool = False) -> CheckResult:
    """存在标题匹配的窗口。

    **遍历所有顶层窗口，不只看前台。** 只看前台会让判定受「任务结束时
    恰好哪个窗口在最上面」影响——那是与任务成败无关的噪声。
    """
    if not IS_WINDOWS:
        return CheckResult(False, "窗口标题判据仅支持 Windows", "window_title")
    try:
        import uiautomation as auto
    except ImportError:
        return CheckResult(False, "uiautomation 未安装，无法做窗口标题判定", "window_title")

    titles: list[str] = []
    try:
        window = auto.GetRootControl().GetFirstChildControl()
        while window:
            name = (window.Name or "").strip()
            if name:
                titles.append(name)
            window = window.GetNextSiblingControl()
    except Exception as exc:  # noqa: BLE001
        return CheckResult(False, f"枚举窗口失败：{exc}", "window_title")

    if regex:
        matcher = re.compile(pattern, re.IGNORECASE)
        hit = next((t for t in titles if matcher.search(t)), "")
    else:
        low = pattern.lower()
        hit = next((t for t in titles if low in t.lower()), "")

    if hit:
        return CheckResult(True, f"窗口标题命中 {hit!r}", "window_title")
    sample = "、".join(titles[:5]) or "（无）"
    return CheckResult(False, f"没有标题含 {pattern!r} 的窗口。当前窗口：{sample}", "window_title")


def check_file_contains(path: str, text: str, encoding: str = "utf-8") -> CheckResult:
    """文件存在且含指定文本。

    **这是最可靠的一类判据**：它检查的是任务真正产生的副作用，
    不受窗口 Z 序、焦点、动画的影响。能设计成文件判定的任务就别用别的。
    """
    target = Path(path)
    if not target.exists():
        return CheckResult(False, f"文件不存在：{target}", "file_contains")
    try:
        content = target.read_text(encoding=encoding, errors="replace")
    except OSError as exc:
        return CheckResult(False, f"读不出 {target}：{exc}", "file_contains")
    if text in content:
        return CheckResult(True, f"{target.name} 含 {text!r}", "file_contains")
    preview = content.strip()[:60].replace("\n", "⏎")
    return CheckResult(
        False, f"{target.name} 不含 {text!r}；实际内容：{preview!r}", "file_contains"
    )


def check_file_exists(path: str, should_exist: bool = True) -> CheckResult:
    target = Path(path)
    exists = target.exists()
    ok = exists if should_exist else not exists
    verb = "应存在" if should_exist else "应不存在"
    return CheckResult(ok, f"{target} {verb}；实际{'存在' if exists else '不存在'}", "file_exists")


#: 判据名 → 实现。任务清单里按名字引用。
CHECKERS = {
    "process": check_process,
    "window_title": check_window_title,
    "file_contains": check_file_contains,
    "file_exists": check_file_exists,
}


# ---------------------------------------------------------------------- #
# 组合
# ---------------------------------------------------------------------- #


@dataclass
class SuccessCheck:
    """一个任务的成功判据，可以由多条组成。

    `mode="all"` 时全部通过才算成功（默认）；`"any"` 时任意一条即可。

    多条判据不是为了保险，是为了**堵住不同的伪成功**。
    比如「打开指定文件」：只查进程，记事本开着但没打开那个文件也算成功；
    只查窗口标题，标题对但可能是别人开的。两条同时要求才排除得掉。
    """

    checks: list[dict] = field(default_factory=list)
    mode: str = "all"

    def run(self) -> tuple[bool, str]:
        """执行全部判据，返回 (是否成功, 可读说明)。"""
        if not self.checks:
            return False, "未定义成功判据——**不能默认算成功**"

        results = []
        for spec in self.checks:
            spec = dict(spec)
            kind = spec.pop("type", "")
            checker = CHECKERS.get(kind)
            if checker is None:
                results.append(CheckResult(False, f"未知判据类型 {kind!r}", kind))
                continue
            try:
                results.append(checker(**spec))
            except TypeError as exc:
                results.append(CheckResult(False, f"{kind} 参数不对：{exc}", kind))
            except Exception as exc:  # noqa: BLE001 —— 判定出错就是判定失败
                results.append(CheckResult(False, f"{kind} 执行出错：{exc}", kind))

        passed = all(results) if self.mode == "all" else any(results)
        lines = [f"{'✓' if r.passed else '✗'} {r.detail}" for r in results]
        return passed, "; ".join(lines)

    @classmethod
    def from_spec(cls, spec: Any) -> SuccessCheck:
        """从任务清单里的 `success_check` 字段构造。

        支持三种写法：单条 dict、多条 list、带 mode 的 dict。
        """
        if isinstance(spec, dict) and "checks" in spec:
            return cls(checks=list(spec["checks"]), mode=spec.get("mode", "all"))
        if isinstance(spec, dict):
            return cls(checks=[spec])
        if isinstance(spec, list):
            return cls(checks=list(spec))
        return cls(checks=[])


__all__ = [
    "CHECKERS",
    "CheckResult",
    "SuccessCheck",
    "check_file_contains",
    "check_file_exists",
    "check_process",
    "check_window_title",
]
