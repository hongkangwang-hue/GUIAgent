"""动作生成评测 —— 大纲 W5「对比微调前后模型在 GUI 任务理解与动作生成上的效果」。

## 判什么

一条样本给「截图 + 任务目标」，模型输出「动作类型 + 坐标」。判两件事：

1. **动作类型对不对**（`left_click` / `double_click` / `mouse_move`）
2. **坐标偏多远**

## 为什么不设单一的「成功率」

动作生成不像 grounding 那样有一个干净的判据。grounding 有真值 bbox，
点落进去就是对；而这里只有一个真值坐标点，**「差多少算对」是人定的**。

所以不报单一数字，报三组：

- **动作类型准确率** —— 无歧义
- **坐标误差的中位数** —— 无阈值，直接看分布
- **联合命中率 @ 半径 r** —— 动作类型对且坐标在 r 内，**r 是显式参数**

一个能随阈值变号的指标不该被当成结论。M2 的稳定性测试栽过这个跟头：
句柄斜率 +9.9 而阈值 10.0，差 0.1 就翻转结论。所以这里把阈值摆在明面上，
并同时报几个不同的 r。

## 按动作类型分开报

训练集里 `left_click` 占 75.9%。**只报总体准确率会被它主导**——
模型全输出 `left_click` 就能拿 76%，而那说明不了任何事。

## 微调前后怎么比

    # 微调前（base 模型零样本）
    python -m eval.action --local Qwen/Qwen2.5-VL-3B-Instruct --tag before

    # 微调后
    python -m eval.action --local Qwen/Qwen2.5-VL-3B-Instruct \\
        --adapter finetune/outputs/<run>/adapter --tag after

    # 汇总对比
    python -m eval.action --report

**两次必须用同一套提示词、同一个坐标空间、同一个验证集。**
微调前的基线必须是自己跑出来的实测值，不能引用官方模型卡数字——
拿官方数字当基线、自测数字当结果，是无效对比。

也支持 API 后端做参照：

    python -m eval.action --provider dashscope --tag qwen-api
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

#: 结果落点，一个模型一个文件。
RESULT_DIR = Path("docs/m3-action")

#: 默认验证集，由 `python -m finetune.dataset` 生成。
VAL_FILE = Path("finetune/data/val.jsonl")

#: 联合命中的半径，归一化坐标单位（空间是 1000×1000）。
#: 50 ≈ 屏幕宽度的 5%，在 1920 宽的屏上约 96px——大致是一个按钮的尺度。
DEFAULT_RADII = (25, 50, 100)


@dataclass
class Prediction:
    sample_id: str
    instruction: str
    truth_action: str = ""
    truth_xy: list = field(default_factory=list)
    pred_action: str = ""
    pred_xy: list = field(default_factory=list)
    action_ok: bool = False
    distance: float = -1.0
    error: str = ""
    latency_ms: float = 0.0
    raw: str = ""


def load_val(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"找不到验证集 {path}。先跑 python -m finetune.dataset")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _done_ids(path: Path) -> set[str]:
    """已完成的 sample_id。坏行跳过——断电时最后一行天然可能残缺。"""
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["sample_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _screenshot_of(record: dict):
    import cv2
    import numpy as np

    from perception.capture import Screenshot
    from perception.types import BBox

    # np.fromfile 而不是 cv2.imread：后者在中文路径上会静默返回 None
    buffer = np.fromfile(record["image"], dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"读不出图片：{record['image']}")
    height, width = image.shape[:2]
    return Screenshot(image=image, region=BBox(0, 0, width, height), engine="dataset")


def evaluate(
    val_path: Path,
    tag: str,
    provider: str = "",
    model: str = "",
    local_model: str = "",
    adapter: str = "",
    limit: int = 0,
    resume: bool = True,
) -> Path:
    """跑一档评测，逐条追加到 JSONL。"""
    from finetune.train_lora import SYSTEM_PROMPT

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULT_DIR / f"{tag}.jsonl"

    rows = load_val(val_path)
    done = _done_ids(out) if resume else set()
    todo = [r for r in rows if r["sample_id"] not in done]
    if limit:
        todo = todo[:limit]

    predict, closer, label = _build_predictor(provider, model, local_model, adapter, SYSTEM_PROMPT)

    print("=" * 70)
    print("动作生成评测")
    print("=" * 70)
    print(f"  标签      {tag}")
    print(f"  模型      {label}")
    print(f"  验证集    {val_path}    共 {len(rows)}，已完成 {len(done)}，本次 {len(todo)}")
    print(f"  结果      {out}（逐条追加，可断点续跑）")
    print()
    if not todo:
        print("  没有待跑样本。加 --no-resume 可重跑全部。")
        return out

    hits = 0
    try:
        with out.open("a", encoding="utf-8") as handle:
            for index, row in enumerate(todo, start=1):
                record = Prediction(
                    sample_id=row["sample_id"],
                    instruction=row["instruction"][:120],
                    truth_action=row["action"],
                    truth_xy=list(row["point_norm"]),
                )
                started = time.perf_counter()
                try:
                    raw, action, point = predict(row)
                    record.raw = raw[:200]
                    record.pred_action = action
                    if point is not None:
                        record.pred_xy = [point.x, point.y]
                        record.distance = round(math.dist(record.truth_xy, record.pred_xy), 2)
                    record.action_ok = action == row["action"]
                except Exception as exc:  # noqa: BLE001 —— 单样本失败不中断整档
                    record.error = f"{type(exc).__name__}: {exc}"[:200]
                record.latency_ms = round((time.perf_counter() - started) * 1000, 1)

                if record.action_ok and 0 <= record.distance <= DEFAULT_RADII[1]:
                    hits += 1
                handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
                handle.flush()

                if index % 10 == 0 or index == len(todo):
                    print(
                        f"  [{index}/{len(todo)}] 联合命中@50 {hits}/{index} = {hits / index:.1%}"
                    )
    except KeyboardInterrupt:
        print("\n  已中断。已完成的结果都在文件里，重跑会自动续上。")
    finally:
        closer()

    return out


def _build_predictor(provider: str, model: str, local_model: str, adapter: str, system: str):
    """返回 (predict, close, label)。

    **API 与本地两条路共用同一个提示词和同一套解析。** 分成两套的话，
    「微调前 vs 微调后」的差值里会混进提示词与解析差异，那就不是模型的差值了。
    """
    from grounding.local_vlm import parse_point

    def extract(raw: str):
        """从模型输出里抠出 (动作名, 坐标)。"""
        action = ""
        try:
            payload = json.loads(raw.strip())
            if isinstance(payload, dict):
                action = str(payload.get("action", "")).strip()
        except (json.JSONDecodeError, TypeError):
            for name in ("double_click", "right_click", "left_click", "mouse_move"):
                if name in raw:
                    action = name
                    break
        return action, parse_point(raw)

    if local_model:
        import torch
        from PIL import Image
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(local_model)
        net = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            local_model,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        if adapter:
            from peft import PeftModel

            net = PeftModel.from_pretrained(net, adapter)
        net.eval()

        def predict(row: dict):
            image = Image.open(row["image"]).convert("RGB")
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": row["instruction"]},
                    ],
                },
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(text=[text], images=[image], return_tensors="pt").to(net.device)
            with torch.no_grad():
                # 动作生成是确定性任务，不采样
                generated = net.generate(**inputs, max_new_tokens=64, do_sample=False)
            raw = processor.decode(
                generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            action, point = extract(raw)
            return raw, action, point

        return predict, lambda: None, f"本地 {local_model}" + (f" + {adapter}" if adapter else "")

    from llm.openai_compat import OpenAICompatBackend
    from llm.providers import load_dotenv_if_present, resolve

    load_dotenv_if_present()
    config = resolve(provider or "dashscope", model=model or None)
    backend = OpenAICompatBackend(config=config, system_prompt=system)

    def predict(row: dict):
        intent = backend.predict_action(row["instruction"], _screenshot_of(row))
        raw = intent.raw_text or ""
        action = intent.action_type or ""
        point = None
        x, y = intent.params.get("x"), intent.params.get("y")
        if x is not None and y is not None:
            from perception.types import Point

            point = Point(int(x), int(y))
        if point is None:
            _, point = extract(raw)
        return raw, action, point

    return predict, backend.close, f"{config.provider.label} / {config.model}"


def summarize(path: Path, radii: tuple = DEFAULT_RADII) -> dict:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    valid = [r for r in rows if not r["error"]]
    with_xy = [r for r in valid if r["distance"] >= 0]
    distances = sorted(r["distance"] for r in with_xy)

    stats: dict = {
        "tag": path.stem,
        "total": len(rows),
        "errors": len(rows) - len(valid),
        "no_coords": len(valid) - len(with_xy),
        "action_acc": (sum(r["action_ok"] for r in valid) / len(valid)) if valid else 0.0,
        "median_distance": distances[len(distances) // 2] if distances else -1.0,
        "joint": {},
        "by_action": {},
        "median_latency_ms": 0.0,
    }
    for radius in radii:
        hit = sum(1 for r in with_xy if r["action_ok"] and r["distance"] <= radius)
        stats["joint"][str(radius)] = hit / len(valid) if valid else 0.0

    latencies = sorted(r["latency_ms"] for r in rows if r["latency_ms"])
    if latencies:
        stats["median_latency_ms"] = latencies[len(latencies) // 2]

    buckets: dict = defaultdict(lambda: {"n": 0, "action_ok": 0, "dists": []})
    for row in valid:
        bucket = buckets[row["truth_action"]]
        bucket["n"] += 1
        bucket["action_ok"] += row["action_ok"]
        if row["distance"] >= 0:
            bucket["dists"].append(row["distance"])
    for name, bucket in buckets.items():
        dists = sorted(bucket["dists"])
        stats["by_action"][name] = {
            "n": bucket["n"],
            "action_acc": bucket["action_ok"] / bucket["n"],
            "median_distance": dists[len(dists) // 2] if dists else -1.0,
        }
    return stats


def print_summary(stats: dict) -> None:
    print()
    print("=" * 70)
    print(f"汇总 {stats['tag']}")
    print("=" * 70)
    print(f"  样本 {stats['total']}    调用失败 {stats['errors']}    没给坐标 {stats['no_coords']}")
    print(f"  **动作类型准确率 {stats['action_acc']:.1%}**")
    print(f"  坐标误差中位数   {stats['median_distance']:.1f}（归一化 1000 空间）")
    print("  联合命中（动作类型对 且 坐标在半径内）：")
    for radius, value in stats["joint"].items():
        print(f"    r={radius:<5}{value:>7.1%}")
    print(f"  中位延迟 {stats['median_latency_ms']:.0f}ms")
    if stats["by_action"]:
        print(f"\n  {'真值动作':<16}{'样本':>6}{'类型准确率':>12}{'坐标误差中位数':>16}")
        for name, v in sorted(stats["by_action"].items(), key=lambda kv: -kv[1]["n"]):
            print(f"  {name:<16}{v['n']:>6}{v['action_acc']:>12.1%}{v['median_distance']:>16.1f}")
        print("\n  **按动作类型分开看。** 训练集里 left_click 占 75.9%，")
        print("  只报总体准确率会被它主导——全输出 left_click 就能拿 76%。")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="动作生成评测（大纲 W5 微调前后对比）")
    parser.add_argument("--val", default=str(VAL_FILE))
    parser.add_argument("--tag", default="", help="这一档的名字，作为结果文件名")
    parser.add_argument("--provider", default="", help="走 API 后端，如 dashscope")
    parser.add_argument("--model", default="", help="覆盖平台默认模型")
    parser.add_argument("--local", default="", help="走本地模型，给 HF 模型名或路径")
    parser.add_argument("--adapter", default="", help="LoRA adapter 目录，微调后评测用")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--report", action="store_true", help="只汇总已有结果，不调模型")
    args = parser.parse_args()

    if args.report:
        files = sorted(RESULT_DIR.glob("*.jsonl"))
        if not files:
            print(f"{RESULT_DIR} 下没有结果文件。")
            return 1
        for path in files:
            print_summary(summarize(path))
        if len(files) >= 2:
            print()
            print("=" * 70)
            print("对比")
            print("=" * 70)
            rows = [summarize(p) for p in files]
            print(f"  {'档':<24}{'类型准确率':>12}{'误差中位数':>12}{'联合@50':>10}")
            for r in rows:
                print(
                    f"  {r['tag']:<24}{r['action_acc']:>12.1%}"
                    f"{r['median_distance']:>12.1f}{r['joint'].get('50', 0):>10.1%}"
                )
        return 0

    if not args.provider and not args.local:
        raise SystemExit("要么 --provider（API），要么 --local（本地模型）")
    tag = args.tag or (args.provider or Path(args.local).name)

    out = evaluate(
        val_path=Path(args.val),
        tag=tag,
        provider=args.provider,
        model=args.model,
        local_model=args.local,
        adapter=args.adapter,
        limit=args.limit,
        resume=not args.no_resume,
    )
    print_summary(summarize(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
