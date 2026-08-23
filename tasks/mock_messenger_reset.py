"""重置消息模拟程序的状态 —— 基础任务 4 每轮之前跑。

做两件事：清空日志文件，把消息窗口提到前台。

**不重启程序**：`mock_messenger.py` 由人在另一个窗口里保持运行，
评测过程中不该被关掉。重启会改变窗口位置与大小，而那是 Agent 每轮
面对的初始条件之一。

## 为什么要提到前台

第一次 5×5 实测里这个任务 0/5。轨迹显示 Agent 从头到尾在操作**记事本**
——上一个任务留下的窗口压在最上面，消息程序被盖住了。

「窗口在前台」是一个**声明出来的初始条件**，不是给 Agent 放水：
任务是「在测试消息程序里发一条消息」，程序可见是这个任务成立的前提，
就像「关闭计算器」要求计算器先开着一样。真正的难点在于识别输入框、
打字、点发送，那三步一步没少。

反过来说，若把「从一堆窗口里找出目标程序」也算进来，测的就是两件事
混在一起的结果，失败了分不清是哪一环。M4 若要单独测「窗口切换」，
那该是一个独立任务。
"""

from __future__ import annotations

import sys
from pathlib import Path

TITLE = "测试消息"

LOG = (
    Path("C:/agent-test/messages.log")
    if sys.platform == "win32"
    else Path("/tmp/agent-test/messages.log")
)


def activate_window() -> str:
    """把消息窗口提到前台。失败只报告，不中断——起点检查会兜住。"""
    if sys.platform != "win32":
        return "非 Windows，跳过"
    try:
        import uiautomation as auto
    except ImportError:
        return "uiautomation 未安装，跳过"

    try:
        window = auto.GetRootControl().GetFirstChildControl()
        while window:
            if TITLE in (window.Name or ""):
                window.SetActive()
                return f"已激活 {window.Name!r}"
            window = window.GetNextSiblingControl()
    except Exception as exc:  # noqa: BLE001
        return f"激活失败：{exc}"
    return f"没找到标题含 {TITLE!r} 的窗口——消息程序开着吗？"


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    print(activate_window())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
