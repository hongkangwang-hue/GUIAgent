"""重试探针 —— 「多试几次」到底是不是「同一个错误做了 N 遍」。

## 为什么要问这个

§10.5 拆出来：离线-微调组每个子任务磨 **5.9 步**（在线组 3.2 步），
而两组总步数几乎相同（302 vs 294），成功数差 6 倍。**C 不缺步数，
它的步数不产生进展。**

M4 任务 2 要做「错误检测与自动重试」。但重试能不能涨，取决于一件事：

    每次尝试是不是**真的不一样**。

如果模型每次都吐出同一个坐标，重试 6 次就是把同一个错误做 6 遍——
那么 §10 里那个「27.5% 重试 6 次 → 85.5%」的粗算**一分都不成立**，
而 M4 任务 2 的设计要完全不同（要引入多样性，而不只是加重试次数）。

## 第二个用途：查各轮跑在什么屏幕上

报告 §7.1 声称「三组的差别只有一处」，列了模型、部署方式、数据边界——
**但从没验证过屏幕分辨率是否一致**。而 `docs/m2-runs/*.json` 存档里
**没有这一项**，只有轨迹的 `meta.json` 记了 `environment.resolution`
与 DPI 缩放。

§9 刚证明分辨率值 2 倍坐标误差。**若 A/B/C 三组跑在不同分辨率上，
§7 的比较就不是单变量的**，而这一点在存档里完全看不出来。

所以探针顺带把每条轨迹的分辨率、DPI 缩放、后端、模型一起导出。
出现两种以上分辨率会显式报警。

## 为什么不直接把轨迹拷过来

Agent 跑在客机里，`outputs/` 是 gitignore 的。而轨迹里带着
`frames/*.png`——**整屏截图**。

M2 记录过两次真实事故：Agent 改写并保存了 `.env`、对着真实微软账号
登录框按了 12 次回车。**那两次的截图里就有这些东西。**
`model_thinking` 与 `raw_output` 是模型对屏幕的描述，同样可能带出来。

所以这个脚本**默认只导出结构化的动作数据**：动作类型、坐标、执行状态。
自由文本要 `--include-text` 显式打开，且逐条截断。

## 用法（在**客机**里跑）

    python scripts/probe_retries.py
    python scripts/probe_retries.py --out docs/m3-retry-probe.json

产物是一个小 JSON，可以直接提交，然后在宿主机上分析。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TRAJ_DIR = Path("outputs/trajectories")
DEFAULT_OUT = Path("docs/m3-retry-probe.json")

#: 坐标差在这个半径内就算「同一次尝试」。GUI 按钮几十像素宽，
#: 差几个像素点的是同一个东西——只按逐字相等算会低估重复率。
SAME_SPOT_RADIUS = 10.0

#: 自由文本的截断长度。开了 --include-text 也不整段导出。
TEXT_CAP = 80


def load_steps(traj_dir: Path) -> list[dict]:
    path = traj_dir / "steps.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # 断电时最后一行天然可能残缺
    return out


def action_of(step: dict) -> tuple[str, float | None, float | None]:
    """(动作类型, x, y)，全部取**模型坐标系**的值。

    不用 `action_real_coords`：那是换算到客机屏幕之后的，两次模型输出
    相同、缩放系数不同的话会被算成两次不同尝试。**我们要问的是模型
    有没有改主意，不是像素落在哪。**
    """
    intent = step.get("action_intent") or {}
    params = intent.get("params") or {}
    coords = step.get("action_model_coords") or {}
    kind = intent.get("action_type") or coords.get("action") or ""
    x = params.get("x", coords.get("x"))
    y = params.get("y", coords.get("y"))
    return str(kind), x, y


def same_attempt(a: tuple, b: tuple) -> bool:
    """两次尝试算不算「同一次」。"""
    if a[0] != b[0]:
        return False
    if a[1] is None or b[1] is None:
        return a[1] is None and b[1] is None
    return math.dist((a[1], a[2]), (b[1], b[2])) <= SAME_SPOT_RADIUS


def analyse_subtask(steps: list[dict], include_text: bool, include_subtasks: bool = False) -> dict:
    attempts = [action_of(s) for s in steps]

    # 去重：逐个跟已有的比，落在 SAME_SPOT_RADIUS 内就并进去
    distinct: list[tuple] = []
    for attempt in attempts:
        if not any(same_attempt(attempt, seen) for seen in distinct):
            distinct.append(attempt)

    repeats = Counter()
    for attempt in attempts:
        for seen in distinct:
            if same_attempt(attempt, seen):
                repeats[str(seen)] += 1
                break
        else:  # pragma: no cover —— distinct 由 attempts 构造，不该走到
            repeats[str(attempt)] += 1

    row = {
        "subtask_id": steps[0].get("subtask_id"),
        "subtask": str(steps[0].get("subtask", ""))[:TEXT_CAP] if include_subtasks else None,
        "steps": len(steps),
        "distinct_attempts": len(distinct),
        "max_repeat": max(repeats.values()) if repeats else 0,
        "statuses": dict(Counter(s.get("execution_status", "") for s in steps)),
        "attempts": [{"action": k, "x": x, "y": y} for k, x, y in attempts],
    }
    if include_text:
        row["thinking"] = [str(s.get("model_thinking", ""))[:TEXT_CAP] for s in steps]
    return row


def environment_of(traj_dir: Path) -> dict:
    """轨迹跑在什么屏幕上。**存档里没有这一项，只有这里有。**

    分辨率与 DPI 缩放一起取：2560×1600 @150% 和 1024×768 @100% 下，
    同一个按钮在截图里的像素尺寸差 2.5 倍——**只记分辨率不记缩放，
    仍然说不清模型看到的元素有多大**。
    """
    path = traj_dir / "meta.json"
    if not path.exists():
        return {}
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    env = meta.get("environment") or {}
    dpi = env.get("dpi") or {}
    return {
        "resolution": env.get("resolution", ""),
        "scale_factor": dpi.get("scale_factor"),
        "capture_engine": env.get("capture_engine", ""),
        "backend": meta.get("backend", ""),
        "model": meta.get("model", ""),
        "started_at": meta.get("started_at", ""),
        "status": meta.get("status", ""),
    }


def analyse(traj_dir: Path, include_text: bool, include_subtasks: bool = False) -> dict | None:
    steps = load_steps(traj_dir)
    if not steps:
        return None

    by_subtask: dict = {}
    for step in steps:
        by_subtask.setdefault(step.get("subtask_id"), []).append(step)

    subtasks = [
        analyse_subtask(group, include_text, include_subtasks) for group in by_subtask.values()
    ]
    total = sum(s["steps"] for s in subtasks)
    distinct = sum(s["distinct_attempts"] for s in subtasks)
    return {
        "trajectory_id": traj_dir.name,
        "env": environment_of(traj_dir),
        "steps": total,
        "subtasks": len(subtasks),
        "distinct_attempts": distinct,
        # **这就是要的那个数。** 1.0 表示每一步都在重复之前试过的东西。
        "repeat_rate": round(1 - distinct / total, 4) if total else None,
        "detail": subtasks,
    }


def summarize(runs: list[dict]) -> dict:
    steps = sum(r["steps"] for r in runs)
    distinct = sum(r["distinct_attempts"] for r in runs)
    multi = [s for r in runs for s in r["detail"] if s["steps"] > 1]

    def screen_of(run: dict) -> str:
        """屏幕设置的标签。**「没记录」要和「不一样」分开。**

        早期轨迹的 meta.json 没有 `environment` 段。把它算成一种「不同的
        屏幕」会误报，而误报会让真的不一致被当成噪声忽略掉。
        """
        env = run.get("env") or {}
        if not env.get("resolution"):
            return "未记录"
        scale = env.get("scale_factor")
        return f"{env['resolution']} @{scale}x" if scale else env["resolution"]

    screens = Counter(screen_of(r) for r in runs)
    known = {k: v for k, v in screens.items() if k != "未记录"}
    return {
        "trajectories": len(runs),
        "screens": dict(screens),
        # **多于一种就意味着 §7 的比较不是单变量的。**
        # 「未记录」不参与判定——那是缺数据，不是不一致。
        "screen_consistent": len(known) <= 1,
        "screens_unknown": screens.get("未记录", 0),
        "steps": steps,
        "distinct_attempts": distinct,
        "repeat_rate": round(1 - distinct / steps, 4) if steps else None,
        "subtasks_with_retries": len(multi),
        "subtasks_where_every_retry_identical": sum(
            1 for s in multi if s["distinct_attempts"] == 1
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="重试探针：每次尝试是不是真的不一样")
    parser.add_argument("--traj-dir", default=str(TRAJ_DIR))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--since", default="", help="只看 id 不早于此的轨迹，如 traj-20260825")
    parser.add_argument(
        "--include-subtasks",
        action="store_true",
        help="导出子任务文本（各截断 80 字）。**比 --include-text 安全**："
        "子任务是规划器对**我们自己那句指令**的拆解,不是模型对屏幕内容的描述。"
        "查「规划器有没有改口」时用",
    )
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="连子任务描述与 thinking 一起导出（各截断 80 字）。"
        "**默认关闭**：那是模型对屏幕的描述，可能把客机屏幕上的内容带出来",
    )
    return parser


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args()
    root = Path(args.traj_dir)
    if not root.exists():
        raise SystemExit(f"{root} 不存在。这个脚本要在**客机**里跑——轨迹在那边。")

    dirs = sorted(d for d in root.iterdir() if d.is_dir() and d.name >= args.since)
    runs = [r for d in dirs if (r := analyse(d, args.include_text, args.include_subtasks))]
    if not runs:
        raise SystemExit(f"{root} 下没有可读的 steps.jsonl")

    stats = summarize(runs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"summary": stats, "runs": runs}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

    print("=" * 70)
    print("重试探针")
    print("=" * 70)
    print(f"  轨迹 {stats['trajectories']}    总步数 {stats['steps']}")
    print(f"  不同的尝试 {stats['distinct_attempts']}    **重复率 {stats['repeat_rate']:.1%}**")
    print(
        f"  有重试的子任务 {stats['subtasks_with_retries']}    "
        f"其中**每次尝试都一样**的 {stats['subtasks_where_every_retry_identical']}"
    )
    print()
    print("  屏幕：")
    for screen, count in stats["screens"].items():
        print(f"    {screen:<24} {count} 条轨迹")
    if stats["screens_unknown"]:
        print(f"    （其中 {stats['screens_unknown']} 条的 meta.json 没有 environment 段）")
    if not stats["screen_consistent"]:
        print()
        print("  ⚠️ **出现了不止一种屏幕设置。** 报告 §7.1 声称「三组的差别只有")
        print("     一处」，那个说法在这批数据上不成立——§9 已证明分辨率值 2 倍")
        print("     坐标误差，跨分辨率的成功率不可直接比较。")
    print()
    print("  重复率高 → 重试只是把同一个错误做 N 遍，M4 任务 2 必须引入")
    print("            多样性（换温度 / 换提示 / 换策略），加次数没用。")
    print("  重复率低 → 模型确实在改主意，加重试次数本身就有价值。")
    print()
    print(f"  写入 {out}    （不含截图；自由文本默认不导出）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
