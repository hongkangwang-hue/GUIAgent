"""从轨迹里统计 Agent 的行为模式 —— M2 交付物 10 与 M4 错误分类的数据源。

## 为什么要单写一个

`run_basic_tasks.py` 记的是**结果**：成功没成功、几步、多久。但「为什么
失败」在结果里看不出来。实测里最刺眼的两个现象都只存在于步与步之间：

- **零动作报完成**：子任务以 `done` 收尾，一个动作都没执行成功
- **重复动作**：同一个动作连发若干次，屏幕上明明弹着「文件名无效」

第二个是人在旁边看着才发现的——文件名框里的路径被拼了八遍。这种东西
不统计出来，报告里只能写成一句印象式的"模型有时会重复操作"，没有分量。

## 只读，不改

扫 `outputs/trajectories/`，不碰任何文件。可以在评测跑完之后任何时候跑。

    python scripts/analyze_trajectories.py
    python scripts/analyze_trajectories.py --since 20260823
    python scripts/analyze_trajectories.py --json docs/m2-behavior-stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _action_key(intent: dict) -> str:
    """把一个动作压成可比较的字符串。

    **坐标要参与比较，但要容差。** 模型重复点同一个按钮时坐标往往差几个
    像素，逐像素比会把重复动作漏掉；完全忽略坐标又会把"点了两个不同按钮"
    误判成重复。取 20 像素的格子是个折中——比按钮小，比抖动大。
    """
    if not intent:
        return ""
    if intent.get("done"):
        return "done"
    parts = [str(intent.get("action") or "")]
    if intent.get("text"):
        parts.append(f"text={intent['text']}")
    if intent.get("keys"):
        parts.append(f"keys={intent['keys']}")
    x, y = intent.get("x"), intent.get("y")
    if x is not None and y is not None:
        parts.append(f"@{int(x) // 20},{int(y) // 20}")
    return "|".join(parts)


def analyze(root: Path, since: str) -> dict:
    from core.trajectory import TrajectoryReader, list_trajectories

    stats = {
        "trajectories": 0,
        "steps": 0,
        "subtasks": 0,
        "empty_done_subtasks": 0,
        "repeat_steps": 0,
        "max_repeat_run": 0,
        "actions": Counter(),
        "execution_status": Counter(),
        "grounding_source": Counter(),
        "by_instruction": defaultdict(lambda: {"n": 0, "steps": 0, "repeats": 0}),
        "worst": [],
    }

    for directory in list_trajectories(root):
        if since and directory.name < f"traj-{since}":
            continue
        try:
            reader = TrajectoryReader(directory)
            meta = reader.meta
            steps = list(reader.iter_steps())
        except Exception as exc:  # noqa: BLE001 —— 坏轨迹跳过，不中断统计
            print(f"  [跳过] {directory.name}: {exc}")
            continue
        if not steps:
            continue

        stats["trajectories"] += 1
        stats["steps"] += len(steps)

        bucket = stats["by_instruction"][meta.instruction[:24]]
        bucket["n"] += 1
        bucket["steps"] += len(steps)

        # --- 逐子任务 ---
        by_sub: dict[int, list] = defaultdict(list)
        for record in steps:
            by_sub[record.subtask_id].append(record)
        for group in by_sub.values():
            stats["subtasks"] += 1
            if group[-1].execution_status == "no_action" and not any(
                r.execution_status == "ok" for r in group
            ):
                stats["empty_done_subtasks"] += 1

        # --- 相邻重复动作 ---
        # 只算**连续**重复。隔着别的动作再回来点同一个地方是正常的
        # （比如点输入框 → 打字 → 再点它），连着发才是没看反馈。
        run = 1
        previous = ""
        for record in steps:
            key = _action_key(record.action_intent)
            stats["actions"][record.action_intent.get("action") or "done"] += 1
            stats["execution_status"][record.execution_status] += 1
            source = (record.grounding or {}).get("source") or "-"
            stats["grounding_source"][source] += 1

            if key and key == previous:
                run += 1
                stats["repeat_steps"] += 1
                bucket["repeats"] += 1
                stats["max_repeat_run"] = max(stats["max_repeat_run"], run)
                if run >= 4:
                    stats["worst"].append(
                        {
                            "trajectory": meta.trajectory_id,
                            "instruction": meta.instruction[:30],
                            "step": record.step,
                            "run": run,
                            "action": key[:60],
                        }
                    )
            else:
                run = 1
            previous = key

    return stats


def main() -> int:
    _console()
    parser = argparse.ArgumentParser(description="从轨迹统计 Agent 行为模式")
    parser.add_argument("--root", default="outputs/trajectories")
    parser.add_argument("--since", default="", help="只看这个日期之后的，如 20260823")
    parser.add_argument("--json", default="", help="同时写一份 JSON")
    args = parser.parse_args()

    stats = analyze(Path(args.root), args.since)
    if not stats["trajectories"]:
        print(f"{args.root} 下没有可读的轨迹")
        return 1

    print("=" * 70)
    print("轨迹行为统计")
    print("=" * 70)
    print(f"  轨迹 {stats['trajectories']} 条    步 {stats['steps']}    子任务 {stats['subtasks']}")
    print()

    empty = stats["empty_done_subtasks"]
    subs = stats["subtasks"]
    print(f"  零动作报完成   {empty}/{subs} 个子任务（{empty / subs:.0%}）")
    print("     子任务以 done 收尾，一个动作都没执行成功。")
    print()

    rep = stats["repeat_steps"]
    print(f"  连续重复动作   {rep}/{stats['steps']} 步（{rep / stats['steps']:.0%}）")
    print(f"     最长连发 {stats['max_repeat_run']} 次。这些步全是无效的：")
    print("     模型不看上一步的结果，也不看屏幕上的错误提示。")
    print()

    print("  动作分布：", dict(stats["actions"].most_common()))
    print("  执行结果：", dict(stats["execution_status"].most_common()))
    print("  定位来源：", dict(stats["grounding_source"].most_common()))
    print()

    print(f"  {'指令':<26}{'轨迹':>6}{'均步':>8}{'重复步占比':>12}")
    print("-" * 70)
    for name, b in sorted(stats["by_instruction"].items(), key=lambda kv: -kv[1]["steps"]):
        print(
            f"  {name:<26}{b['n']:>6}{b['steps'] / b['n']:>8.1f}"
            f"{b['repeats'] / max(b['steps'], 1):>11.0%}"
        )

    if stats["worst"]:
        print()
        print(f"  连发 4 次以上的片段（{len(stats['worst'])} 处，列前 8）：")
        for w in stats["worst"][:8]:
            print(f"    {w['instruction']:<26} 第{w['step']}步 连发{w['run']}次  {w['action']}")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(stats)
        payload["actions"] = dict(stats["actions"])
        payload["execution_status"] = dict(stats["execution_status"])
        payload["grounding_source"] = dict(stats["grounding_source"])
        payload["by_instruction"] = dict(stats["by_instruction"])
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
