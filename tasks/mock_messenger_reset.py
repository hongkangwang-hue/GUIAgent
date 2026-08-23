"""重置消息模拟程序的状态 —— 基础任务 4 每轮之前跑。

只清空日志文件，**不重启程序**：`mock_messenger.py` 由人在另一个窗口
里保持运行，评测过程中不该被关掉。日志清空后界面上的历史记录还在，
但判定读的是文件，两者不一致不影响判定。

> 之所以不顺带重启程序：重启会改变窗口位置与 Z 序，而那是 Agent 每轮
> 面对的初始条件之一。保持程序常驻，每轮的起点才真的一致。
"""

from __future__ import annotations

import sys
from pathlib import Path

LOG = (
    Path("C:/agent-test/messages.log")
    if sys.platform == "win32"
    else Path("/tmp/agent-test/messages.log")
)


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
