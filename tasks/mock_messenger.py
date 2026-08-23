"""本地消息模拟程序 —— M2 基础任务 4「发送消息」的操作对象。

## 为什么要自己写一个

大纲的 5 个基础任务里有「发送消息」。用真实的聊天软件会同时违反两条
M0 全局约束：需要登录真实账号（数据边界），且会向真人发消息（不可撤销
的对外动作）。

所以造一个：界面形态和常见聊天窗口一致（输入框 + 发送按钮 + 消息列表），
但**发出去的消息只写进本地文件**。

## 关键设计：每条消息落盘

`send_log` 是这个任务**唯一的成功判据**。M2 任务 8 要求成功判定必须
程序化，而文件内容是最可靠的一类判据——它检查任务真正产生的副作用，
不受窗口 Z 序、焦点、动画影响。

反过来说：如果只看「消息列表里出现了那句话」，就得靠截图或 UIA 去读
界面，那既慢又受渲染时机影响。落盘让判定变成一次 `read_text()`。

## 无障碍属性是刻意设置的

输入框和按钮都设了明确的可访问名称，这样 UIA 通道能读到它们。
**这不是作弊**——真实的 Windows 应用（记事本、设置、资源管理器）
同样暴露这些属性，M1 的召回率实验已经实测确认。刻意做一个 UIA
读不到的界面反而不真实。

## 用法

    python tasks/mock_messenger.py                    # 默认日志 C:/agent-test/messages.log
    python tasks/mock_messenger.py --log D:/x.log
    python tasks/mock_messenger.py --reset            # 清空日志后启动
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

DEFAULT_LOG = (
    Path("C:/agent-test/messages.log")
    if sys.platform == "win32"
    else Path("/tmp/agent-test/messages.log")
)


class Messenger:
    def __init__(self, log_path: Path) -> None:
        import tkinter as tk
        from tkinter import scrolledtext

        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self.root = tk.Tk()
        self.root.title("测试消息 - Test Messenger")
        self.root.geometry("560x420")

        header = tk.Label(
            self.root,
            text="本地测试用消息程序（消息只写入本地文件，不联网）",
            fg="#666",
            anchor="w",
            padx=10,
            pady=6,
        )
        header.pack(fill="x")

        self.history = scrolledtext.ScrolledText(
            self.root, height=16, state="disabled", font=("Microsoft YaHei", 10)
        )
        self.history.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        row = tk.Frame(self.root)
        row.pack(fill="x", padx=10, pady=(0, 12))

        self.entry = tk.Entry(row, font=("Microsoft YaHei", 11))
        self.entry.pack(side="left", fill="x", expand=True, ipady=6)
        # 可访问名称：UIA 通道靠它识别这是哪个控件
        self.entry.configure(name="消息输入框")
        self.entry.bind("<Return>", lambda _event: self.send())

        self.button = tk.Button(
            row, text="发送", width=10, command=self.send, font=("Microsoft YaHei", 10)
        )
        self.button.pack(side="left", padx=(8, 0), ipady=2)

        self.status = tk.Label(self.root, text="就绪", fg="#888", anchor="w", padx=10)
        self.status.pack(fill="x", pady=(0, 6))

        self.entry.focus_set()
        self._load_existing()

    def _load_existing(self) -> None:
        """启动时把已有日志显示出来，界面与文件保持一致。"""
        if not self.log_path.exists():
            return
        try:
            content = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        self.history.configure(state="normal")
        self.history.insert("end", content)
        self.history.configure(state="disabled")
        self.history.see("end")

    def send(self) -> None:
        text = self.entry.get().strip()
        if not text:
            self.status.configure(text="消息为空，未发送", fg="#c00")
            return

        stamp = dt.datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {text}\n"

        # **先落盘再更新界面。** 判定读的是文件，界面只是给人看的；
        # 反过来的话，界面显示成功但写盘失败会造成假成功。
        with self.log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)
            handle.flush()

        self.history.configure(state="normal")
        self.history.insert("end", line)
        self.history.configure(state="disabled")
        self.history.see("end")

        self.entry.delete(0, "end")
        self.status.configure(text=f"已发送：{text}", fg="#080")

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    parser = argparse.ArgumentParser(description="本地消息模拟程序（M2 基础任务 4）")
    parser.add_argument("--log", default=str(DEFAULT_LOG), help="消息日志路径")
    parser.add_argument("--reset", action="store_true", help="启动前清空日志")
    args = parser.parse_args()

    log_path = Path(args.log)
    if args.reset:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")
        print(f"已清空 {log_path}")

    Messenger(log_path).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
