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

#: 「最近一次」的固定路径，方便脚本引用。
RAW = Path("docs/m2-basic-tasks-raw.json")
#: **每次跑都另存一份带时间戳的。**
#:
#: 原来只写 RAW，于是一次 `--only send_message` 就把前面整整 25 轮的
#: 数据覆盖没了 —— 那 25 轮跑了二十多分钟、花了真实 API 费用，
#: 控制台输出滚过去就再也拿不回来。
#:
#: 评测数据是**跑一次就贵一次**的东西，默认行为不该是覆盖。
RUNS = Path("docs/m2-runs")


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
    #: 起点是否成功建立。False 表示这一轮**无效**，既不算成功也不算失败。
    precondition_ok: bool = True
    precondition_detail: str = ""
    #: 子任务总数，与其中「一个动作都没执行就报完成」的个数。
    subtasks: int = 0
    empty_done: int = 0
    #: 事后剔除标记。**只标记，不删除**——删掉的话没人知道这一批
    #: 从 5 轮变成了 4 轮。目前唯一的用途是记录人为干预：评测期间
    #: 有人碰了鼠标，那一轮的起点被改过，既不算成功也不算失败。
    excluded: bool = False
    exclusion_reason: str = ""


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
        pre = SuccessCheck.from_spec(task.get("precondition")) if task.get("precondition") else None
        max_steps = args.max_steps or task.get("max_steps", 12)
        print(f"── {task['title']}（{task['name']}）  每子任务上限 {max_steps} 步")

        for attempt in range(1, args.repeats + 1):
            print(f"   第 {attempt}/{args.repeats} 次")
            run_reset(task.get("reset"), dry_run=not args.execute)

            record = RunRecord(task=task["name"], title=task["title"], attempt=attempt)

            # **起点检查在 reset 之后、Agent 之前。**
            # reset 不一定成功，而「关闭应用」的判据是「进程应已退出」——
            # 计算器没启动时它会直接打勾，Agent 什么都没做也算成功。
            # 起点没建立的轮次记 invalid，不进成功率的分子也不进分母：
            # 那是环境的失败，混进 Agent 的成功率里两个数字都不可信。
            if pre is not None and args.execute:
                record.precondition_ok, record.precondition_detail = pre.run()
                if not record.precondition_ok:
                    print(f"      [无效轮] 起点未建立：{record.precondition_detail[:120]}")
                    print("        本轮不计入成功率。检查 reset 命令与测试环境。")
                    records.append(record)
                    continue

            backend = OpenAICompatBackend(config)
            session = Session(
                backend,
                NativeGrounding(*space),
                executor,
                capturer,
                config=SessionConfig(loop=LoopConfig(max_iterations=max_steps)),
            )

            started = time.perf_counter()
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

                # **零动作报完成**：子任务以 done 收尾，但一个动作都没执行成功。
                # 实测里这是失败任务的主要形态——模型的 thinking 甚至写着
                # 「截图中未显示任何测试消息程序或发送按钮」，然后输出
                # {"done": true}。Loop 收到 done 就切下一个子任务，于是
                # 6 个子任务「全部成功」、整条轨迹 status=success，而任务
                # 实际什么都没做。
                #
                # M2 是基线，不在这里加闸拦它——那是 M3 该做的改进，
                # 提前塞进基线里 M3 就没有可比的对照。但**必须量出来**，
                # 否则 M4 的错误分类只能靠人翻轨迹。
                record.subtasks = len(result.outcomes)
                record.empty_done = sum(
                    1
                    for outcome in result.outcomes
                    if getattr(outcome.result, "status", "") == "done"
                    and not any(
                        getattr(step, "execution_status", "") == "ok"
                        for step in getattr(outcome.result, "records", [])
                    )
                )
            except Exception as exc:  # noqa: BLE001 —— 单轮崩溃不该中断整批
                record.error = f"{type(exc).__name__}: {exc}"[:200]
                print(f"      [异常] {record.error}")

            record.duration_s = round(time.perf_counter() - started, 1)

            # **判定必须在 reset 之前、执行之后。** 顺序错了就什么都查不到。
            if args.execute:
                time.sleep(1.5)  # 等界面稳定，避免拿到中间态
                record.verified, record.verify_detail = check.run()
            else:
                # 演练也把判据跑一遍，但**不计入成功**。
                # 判据全是只读的（查进程、查窗口标题、读文件），跑一遍没有副作用；
                # 而 YAML 里判据写错（类型名拼错、参数对不上、路径写错）如果要等到
                # 实机 25 轮跑完才发现，那 25 轮就白跑了。这里让它当场暴露。
                _, detail = check.run()
                record.verified = False
                record.verify_detail = f"[演练不计分] 判据当前状态：{detail}"

            mark = "✓" if record.verified else "✗"
            print(
                f"      {mark} 判定{'通过' if record.verified else '未通过'}"
                f"  步数 {record.steps}  用时 {record.duration_s}s"
                f"  模型自报完成={record.model_said_done}"
            )
            if not record.verified:
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

    # **无效轮与剔除轮都不进分母。** 起点没建立、或人为干预过的轮次
    # 既不是成功也不是失败，把它当失败会低估能力，当成功会高估，
    # 只有排除掉这个数字才有意义。
    for group in by_task.values():
        valid = [r for r in group if r.precondition_ok and not r.excluded]
        if not valid:
            print(f"{group[0].title:<14}{'全部无效':>10}{'':>8}{'':>10}{'':>10}{'':>10}")
            continue
        ok = sum(r.verified for r in valid)
        said = sum(r.model_said_done for r in valid)
        steps = sum(r.steps for r in valid) / len(valid)
        secs = sum(r.duration_s for r in valid) / len(valid)
        print(
            f"{group[0].title:<14}"
            f"{f'{ok}/{len(valid)}':>10}"
            f"{ok / len(valid):>8.0%}"
            f"{steps:>10.1f}"
            f"{secs:>9.1f}s"
            f"{f'{said}/{len(valid)}':>10}"
        )

    valid_all = [r for r in records if r.precondition_ok and not r.excluded]
    invalid = [r for r in records if not r.precondition_ok]
    dropped = [r for r in records if r.precondition_ok and r.excluded]
    total_ok = sum(r.verified for r in valid_all)
    total_said = sum(r.model_said_done for r in valid_all)
    print("-" * 74)
    if valid_all:
        print(
            f"{'合计':<14}"
            f"{f'{total_ok}/{len(valid_all)}':>10}"
            f"{total_ok / len(valid_all):>8.0%}"
            f"{'':>10}{'':>10}"
            f"{f'{total_said}/{len(valid_all)}':>10}"
        )
    else:
        print("  没有一轮的起点建立成功，全部无效。")

    if invalid:
        print("")
        print(f"  **无效轮 {len(invalid)}/{len(records)}（起点未建立，已排除出成功率）**")
        for r in invalid[:5]:
            print(f"    {r.title} 第{r.attempt}次：{r.precondition_detail[:90]}")

    # 模型自报完成 vs 程序化判定 —— 两者的差就是「模型以为做完了但没做完」
    gap = [r for r in valid_all if r.model_said_done and not r.verified]
    if gap:
        print(f"\n  **模型自报完成但判定未通过：{len(gap)} 次**")
        print("  这个差值是 M4 错误分类的重点素材——模型的自我判断不可信到什么程度，")
        print("  只有程序化判定能量出来。")
        for r in gap[:5]:
            print(f"    {r.title} 第{r.attempt}次：{r.verify_detail[:90]}")

    subtasks = sum(r.subtasks for r in valid_all)
    empty = sum(r.empty_done for r in valid_all)
    if subtasks:
        print("")
        print(f"  **零动作报完成的子任务：{empty}/{subtasks}（{empty / subtasks:.0%}）**")
        print("  子任务以 done 收尾，却一个动作都没执行成功。这是失败任务的主要形态，")
        print("  也是 M3 要改的第一个东西——朴素 Loop 无条件相信模型的 done。")

    if dropped:
        print("")
        print(f"  **事后剔除 {len(dropped)} 轮（已排除出成功率）**")
        for r in dropped:
            print(f"    {r.title} 第{r.attempt}次：{r.exclusion_reason}")

    cost = sum(r.cost_cny for r in records)
    print(f"\n  累计成本 {cost:.4f} 元（{len(records)} 轮）")
    if records:
        print(f"  单任务平均 {cost / len(records):.4f} 元")

    payload = json.dumps(
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "executed": bool(args.execute),
            "repeats": args.repeats,
            "scope": args.only or "all",
            "records": [asdict(r) for r in records],
        },
        ensure_ascii=False,
        indent=2,
    )
    RAW.parent.mkdir(parents=True, exist_ok=True)
    RAW.write_text(payload, encoding="utf-8")

    RUNS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    scope = args.only or "all"
    mode = "exec" if args.execute else "dry"
    archive = RUNS / f"{stamp}-{scope}-{mode}.json"
    archive.write_text(payload, encoding="utf-8")
    print(f"  原始数据 {RAW}")
    print(f"  存档     {archive}")
    if not args.execute:
        print("\n  [演练模式] 未真正执行，成功率无意义。加 --execute 实机跑。")


if __name__ == "__main__":
    raise SystemExit(main())
