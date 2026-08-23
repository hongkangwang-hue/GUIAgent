"""跑 M2 的 5 个基础任务，每个 N 次，程序化判定成功率。

对应 M2 任务 8 与验收标准 2、3、5、6、7、10。

## 一轮的流程

    reset（恢复初始状态）→ 跑 Agent → 程序化判定 → 记录 → 下一轮

**reset 在每一轮之前，不是每个任务之前。** 第 3 次开始时的环境若与
第 1 次不同，这 5 次的成功率就不可比——而那正是本脚本要测的数字。

## 判定不看模型说了什么

模型自报 `done` 只表示循环正常收尾，**不等于任务完成**。成功与否由
`core/verify.py` 查客机的进程 / 窗口 / 文件决定。两者在轨迹里是分开的
字段（`status` 与 `verified`），M4 分析失败原因时要能区分
「模型以为做完了但没做完」和「模型自己也知道没做完」。

## 安全

默认**演练模式**（`dry_run`），不碰键鼠，用于检查任务清单与判定器本身
有没有写错。真正执行要显式加 `--execute`。

    python scripts/run_basic_tasks.py                    # 演练
    python scripts/run_basic_tasks.py --execute          # 实机执行
    python scripts/run_basic_tasks.py --execute --repeats 5
    python scripts/run_basic_tasks.py --execute --only open_browser
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TASK_FILE = Path("tasks/basic_tasks.yaml")
REPORT = Path("docs/m2-basic-tasks-report.md")
RAW = Path("docs/m2-basic-tasks-raw.json")


@dataclass
class RunRecord:
    task: str
    title: str
    attempt: int
    verified: bool = False
    verify_detail: str = ""
    loop_status: str = ""
    model_said_done: bool = False
    steps: int = 0
    duration_s: float = 0.0
    cost_cny: float = 0.0
    trajectory_id: str = ""
    error: str = ""
    latency: dict = field(default_factory=dict)


def _console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def run_reset(commands: list[str], dry_run: bool) -> None:
    """执行 reset 命令。失败不中断——`taskkill` 在进程本就不存在时会返回
    非零，那是正常情况，不是错误。"""
    for command in commands or []:
        if dry_run:
            print(f"      [演练] reset: {command}")
            continue
        subprocess.run(
            command,
            shell=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    if not dry_run and commands:
        time.sleep(1.5)  # 给进程退出与窗口关闭留时间


def latency_breakdown(records: list) -> dict:
    """把每步的四段延迟汇总。M2 验收标准 3 要求按 API / grounding /
    执行 / 截图 分解。"""
    total = {"api_ms": 0.0, "grounding_ms": 0.0, "execute_ms": 0.0, "screenshot_ms": 0.0}
    for record in records:
        for key in total:
            total[key] += float((record.latency or {}).get(key, 0.0))
    return {k: round(v, 1) for k, v in total.items()}


def main() -> int:
    _console()
    parser = argparse.ArgumentParser(description="M2 基础任务批量执行与成功率统计")
    parser.add_argument("--execute", action="store_true", help="真正操作键鼠（默认演练）")
    parser.add_argument("--repeats", type=int, default=5, help="每个任务跑几次")
    parser.add_argument("--only", default="", help="只跑某个任务（按 name）")
    parser.add_argument("--tasks", default=str(TASK_FILE))
    parser.add_argument("--provider", default="dashscope")
    parser.add_argument("--max-steps", type=int, default=0, help="覆盖清单里的 max_steps")
    args = parser.parse_args()

    import yaml

    from agent.session import Session, SessionConfig
    from control.executor import ActionExecutor
    from core.loop import LoopConfig
    from core.verify import SuccessCheck
    from grounding.native import NativeGrounding
    from llm.openai_compat import OpenAICompatBackend
    from llm.providers import load_dotenv_if_present, resolve
    from perception.capture import ScreenCapturer
    from perception.coordinate import CoordinateScaler

    spec = yaml.safe_load(Path(args.tasks).read_text(encoding="utf-8"))
    tasks = spec["tasks"]
    if args.only:
        tasks = [t for t in tasks if t["name"] == args.only]
        if not tasks:
            raise SystemExit(f"任务清单里没有 {args.only!r}")

    load_dotenv_if_present()
    config = resolve(args.provider)

    print("=" * 74)
    print("M2 基础任务批量执行")
    print("=" * 74)
    print(
        f"  任务数    {len(tasks)}    每个跑 {args.repeats} 次    共 {len(tasks) * args.repeats} 轮"
    )
    print(f"  模型      {config.model}")
    print(f"  模式      {'**实机执行**（会真的操作键鼠）' if args.execute else '演练（不碰键鼠）'}")
    if args.execute:
        print("\n  急停：Ctrl+Alt+Q，或把鼠标甩到屏幕角落触发 FAILSAFE")
        print("  执行期间请勿操作鼠标键盘。")
        if input("\n  确认开始？(yes/N) ").strip().lower() not in ("yes", "y"):
            print("  已取消")
            return 1
    print()

    capturer = ScreenCapturer()
    probe = capturer.capture(fresh=True)
    scaler = CoordinateScaler(probe.region)
    space = SessionConfig().coordinate_space
    scaler.register("planner", *space)
    executor = ActionExecutor(scaler, space_name="planner", dry_run=not args.execute)

    records: list[RunRecord] = []
    for task in tasks:
        check = SuccessCheck.from_spec(task.get("success_check"))
        max_steps = args.max_steps or task.get("max_steps", 12)
        print(f"── {task['title']}（{task['name']}）  上限 {max_steps} 步")

        for attempt in range(1, args.repeats + 1):
            print(f"   第 {attempt}/{args.repeats} 次")
            run_reset(task.get("reset"), dry_run=not args.execute)

            backend = OpenAICompatBackend(config)
            session = Session(
                backend,
                NativeGrounding(*space),
                executor,
                capturer,
                config=SessionConfig(loop=LoopConfig(max_iterations=max_steps)),
            )

            started = time.perf_counter()
            record = RunRecord(task=task["name"], title=task["title"], attempt=attempt)
            try:
                result = session.run(task["instruction"])
                record.loop_status = result.status
                record.model_said_done = bool(result.succeeded)
                record.steps = result.total_steps
                record.trajectory_id = result.trajectory_id
                record.cost_cny = round(result.cost.cost_cny, 6)
                # 每个子任务的 LoopResult 里带着该子任务的全部 StepRecord
                all_steps = [
                    step
                    for outcome in result.outcomes
                    for step in getattr(outcome.result, "records", [])
                ]
                record.latency = latency_breakdown(all_steps)
            except Exception as exc:  # noqa: BLE001 —— 单轮崩溃不该中断整批
                record.error = f"{type(exc).__name__}: {exc}"[:200]
                print(f"      [异常] {record.error}")

            record.duration_s = round(time.perf_counter() - started, 1)

            # **判定必须在 reset 之前、执行之后。** 顺序错了就什么都查不到。
            if args.execute:
                time.sleep(1.5)  # 等界面稳定，避免拿到中间态
                record.verified, record.verify_detail = check.run()
            else:
                record.verified, record.verify_detail = False, "演练模式未执行，不判定"

            mark = "✓" if record.verified else "✗"
            print(
                f"      {mark} 判定{'通过' if record.verified else '未通过'}"
                f"  步数 {record.steps}  用时 {record.duration_s}s"
                f"  模型自报完成={record.model_said_done}"
            )
            if not record.verified and args.execute:
                print(f"        {record.verify_detail[:150]}")
            records.append(record)
        print()

    render(records, args)
    return 0


def render(records: list[RunRecord], args) -> None:
    from collections import defaultdict

    by_task: dict[str, list[RunRecord]] = defaultdict(list)
    for r in records:
        by_task[r.task].append(r)

    print("=" * 74)
    print("结果汇总")
    print("=" * 74)
    print(f"{'任务':<14}{'成功率':>10}{'占比':>8}{'平均步数':>10}{'平均耗时':>10}{'模型自报':>10}")
    print("-" * 74)

    for group in by_task.values():
        ok = sum(r.verified for r in group)
        said = sum(r.model_said_done for r in group)
        steps = sum(r.steps for r in group) / len(group)
        secs = sum(r.duration_s for r in group) / len(group)
        print(
            f"{group[0].title:<14}"
            f"{f'{ok}/{len(group)}':>10}"
            f"{ok / len(group):>8.0%}"
            f"{steps:>10.1f}"
            f"{secs:>9.1f}s"
            f"{f'{said}/{len(group)}':>10}"
        )

    total_ok = sum(r.verified for r in records)
    total_said = sum(r.model_said_done for r in records)
    print("-" * 74)
    print(
        f"{'合计':<14}"
        f"{f'{total_ok}/{len(records)}':>10}"
        f"{total_ok / len(records):>8.0%}"
        f"{'':>10}{'':>10}"
        f"{f'{total_said}/{len(records)}':>10}"
    )

    # 模型自报完成 vs 程序化判定 —— 两者的差就是「模型以为做完了但没做完」
    gap = [r for r in records if r.model_said_done and not r.verified]
    if gap:
        print(f"\n  **模型自报完成但判定未通过：{len(gap)} 次**")
        print("  这个差值是 M4 错误分类的重点素材——模型的自我判断不可信到什么程度，")
        print("  只有程序化判定能量出来。")
        for r in gap[:5]:
            print(f"    {r.title} 第{r.attempt}次：{r.verify_detail[:90]}")

    cost = sum(r.cost_cny for r in records)
    print(f"\n  累计成本 {cost:.4f} 元（{len(records)} 轮）")
    if records:
        print(f"  单任务平均 {cost / len(records):.4f} 元")

    RAW.parent.mkdir(parents=True, exist_ok=True)
    RAW.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "executed": bool(args.execute),
                "repeats": args.repeats,
                "records": [asdict(r) for r in records],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  原始数据 {RAW}")
    if not args.execute:
        print("\n  [演练模式] 未真正执行，成功率无意义。加 --execute 实机跑。")


if __name__ == "__main__":
    raise SystemExit(main())
