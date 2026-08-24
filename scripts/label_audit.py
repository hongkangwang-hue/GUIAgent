"""核对已打的错误标签，并给出频次统计。

## 两个用途

1. **查畸形标签。** `label` 命令遇到词表外的标签**不拒绝、只提醒**——
   现场打标时因为一个拼写卡住人不合适。代价是错误会留在数据里：
   实测就出现过粘贴重复造成的 `stuck_loopstuck_loop`。这个脚本把它们
   挑出来，附上修复命令。

2. **频次统计。** M4 任务 1 要「按实测频次重建错误分类体系」：删零频次类、
   合并近似类、补新发现类。那一步的输入就是这张表。

## 用法

    python scripts/label_audit.py
    python scripts/label_audit.py --since 20260824
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="核对错误标签并统计频次")
    parser.add_argument("--root", default="outputs/trajectories")
    parser.add_argument("--since", default="", help="只看这个日期之后的，如 20260824")
    parser.add_argument("--json", default="", help="同时写一份 JSON")
    args = parser.parse_args()

    from core.trajectory import ERROR_LABELS, TrajectoryReader, list_trajectories

    counts: Counter = Counter()
    combos: Counter = Counter()
    unknown: list[dict] = []
    labeled_steps = 0
    labeled_trajs = 0
    by_traj: dict[str, list] = defaultdict(list)

    for directory in list_trajectories(Path(args.root)):
        if args.since and directory.name < f"traj-{args.since}":
            continue
        try:
            reader = TrajectoryReader(directory)
            steps = list(reader.iter_steps())
        except Exception as exc:  # noqa: BLE001
            print(f"  [跳过] {directory.name}: {exc}")
            continue

        hit = False
        for record in steps:
            if not record.labels:
                continue
            hit = True
            labeled_steps += 1
            counts.update(record.labels)
            combos[" + ".join(sorted(record.labels))] += 1
            by_traj[directory.name].append((record.step, record.labels))
            for label in record.labels:
                if label not in ERROR_LABELS:
                    unknown.append(
                        {
                            "trajectory": directory.name,
                            "step": record.step,
                            "label": label,
                            "all": list(record.labels),
                        }
                    )
        if hit:
            labeled_trajs += 1

    print("=" * 70)
    print("错误标签核对")
    print("=" * 70)
    print(f"  已打标 {labeled_trajs} 条轨迹 / {labeled_steps} 步\n")

    if unknown:
        print(f"  **词表外的标签 {len(unknown)} 处** —— 多半是粘贴重复或拼写错：\n")
        for item in unknown:
            good = [x for x in item["all"] if x in ERROR_LABELS]
            guess = ",".join(good) if good else "<填正确标签>"
            print(f"    {item['trajectory']}  #{item['step']}  {item['label']!r}")
            print(
                f"      修：python -m cli label {item['trajectory']} "
                f'--step {item["step"]} --labels {guess} --note "..."'
            )
        print()
    else:
        print("  没有词表外的标签。\n")

    if counts:
        print(f"  {'标签':<18}{'次数':>6}  含义")
        print("-" * 70)
        for label, n in counts.most_common():
            print(f"  {label:<18}{n:>6}  {ERROR_LABELS.get(label, '**不在词表里**')}")
        print()

        # 零频次类别是 M4 重建分类体系时要删掉的那些
        missing = [k for k in ERROR_LABELS if k not in counts]
        if missing:
            print(f"  零频次（M4 重建分类时考虑删除）：{'、'.join(missing)}\n")

        print("  标签组合：")
        for combo, n in combos.most_common():
            print(f"    {n:>3} × {combo}")
        print()

    if args.json and counts:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "trajectories": labeled_trajs,
                    "steps": labeled_steps,
                    "counts": dict(counts),
                    "combos": dict(combos),
                    "unknown": unknown,
                    "by_trajectory": {
                        k: [[step, list(labels)] for step, labels in v] for k, v in by_traj.items()
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  已写入 {out}")

    return 1 if unknown else 0


if __name__ == "__main__":
    raise SystemExit(main())
