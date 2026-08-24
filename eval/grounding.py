"""ScreenSpot 定位准确率评测 —— M3 交付物 6 与验收标准 3 的执行器。

## 判据只有一条

**预测点落入真值 bbox 即为正确。** 不算 IoU、不算距离、不设部分分。

理由是这个指标要对应真实后果：GUI Agent 拿到坐标就去点，点进框里就成，
差一像素也是差。M1 的召回率实验已经踩过一次相关的坑——用 IoU 判定时，
同一批标注因为人框的画法不同，分数在 10% 与 100% 之间跳。
**点落入框内是唯一不受标注风格影响的判据。**

## 坐标空间必须显式声明

模型输出的坐标不一定是图像像素。实测：`qwen3-vl` 输出的是归一化到
[0,1000) 的值，与送多大的图无关；而部分 Qwen2.5-VL 版本输出图像像素。

**搞错不会报错**，只会让所有点系统性错位，看起来极像「模型定位不准」。
所以 `--space` 是显式参数而不是嗅探，默认 `1000x1000`。
**接任何新后端，第一件事是用几个样本确认这个值。**

## ScreenSpot 是零样本测试集

不参与训练、不参与验证、不参与提示词选型、不参与超参调优。
任何一项沾上都构成测试泄漏，M3 全周结论作废。本模块**只读**数据集，
不写任何东西回 `data/`。

## 断点续跑

一档六个模型、每档上千样本，按实测的 API 延迟（智谱付费档连发时
11–44 秒/次）单档要跑几小时。**中途断了不能从头再来**——结果按 JSONL
逐条追加，重跑时自动跳过已完成的 `sample_id`。

这条是被教训出来的：M2 期间评测产物被覆盖过两次，每次都是几十分钟的
真实 API 费用打水漂。评测数据跑一次贵一次，默认行为不该是重来。

## 用法

    python -m eval.grounding --provider dashscope --limit 20      # 冒烟
    python -m eval.grounding --provider dashscope --platform desktop
    python -m eval.grounding --provider zhipu --model glm-4.6v
    python -m eval.grounding --report                              # 只汇总已有结果
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from perception.types import BBox, Point

logger = logging.getLogger(__name__)

#: 结果落点。一个模型一个文件，文件名里带模型名，便于并行跑多档。
RESULT_DIR = Path("docs/m3-grounding")

#: 元素尺寸分档。小目标是 GUI grounding 的公认难点，分档统计才看得出
#: 微调到底改善了哪一段。面积单位是像素²。
SIZE_BUCKETS = (
    ("极小 <1k", 0, 1_000),
    ("小 1k-5k", 1_000, 5_000),
    ("中 5k-20k", 5_000, 20_000),
    ("大 >20k", 20_000, 10**12),
)

#: 定位专用提示词。**不复用 Agent 的执行模板。**
#:
#: 执行模板里有动作空间、历史、子任务约定，那些在这里全是噪声——
#: 评的是「能不能找到这个元素」，不是「该做什么动作」。混用会让
#: 分数同时受提示词工程影响，横评就不再是模型能力的对比。
SYSTEM_PROMPT = """你是一个 GUI 元素定位器。你会看到一张界面截图和一个元素描述。

只输出一个 JSON 对象，不要任何解释文字：

{"action": "left_click", "x": 横坐标, "y": 纵坐标}

坐标指向该元素的中心，原点在图片左上角。"""


@dataclass
class Prediction:
    sample_id: str
    platform: str
    element_kind: str
    instruction: str
    #: 真值框（图像像素）
    bbox: list = field(default_factory=list)
    #: 模型原始坐标与换算后的图像像素坐标
    raw_xy: list = field(default_factory=list)
    pixel_xy: list = field(default_factory=list)
    hit: bool = False
    error: str = ""
    latency_ms: float = 0.0
    tokens: int = 0
    cost_cny: float = 0.0


def to_pixels(x: float, y: float, space: tuple[int, int], size: tuple[int, int]) -> Point:
    """把模型坐标换算成图像像素。

    `space == size` 时是恒等变换——那正是「模型直接输出图像像素」的情形，
    所以这个函数不需要为两种模型分支。
    """
    return Point(
        int(round(x / space[0] * size[0])),
        int(round(y / space[1] * size[1])),
    )


def inside(point: Point, box: BBox) -> bool:
    """点是否落在框内。右/下为开区间，与 `BBox` 的语义一致。"""
    return box.left <= point.x < box.right and box.top <= point.y < box.bottom


def _bucket(area: int) -> str:
    for name, low, high in SIZE_BUCKETS:
        if low <= area < high:
            return name
    return SIZE_BUCKETS[-1][0]


def load_samples(platform: str = "", dataset: str = "screenspot") -> list:
    """读 ScreenSpot 样本。**只读，不写回。**"""
    from data.loaders import LOADERS

    loader_cls = LOADERS.get(dataset)
    if loader_cls is None:
        raise SystemExit(f"没有名为 {dataset!r} 的装载器。可用：{'、'.join(LOADERS)}")
    loader = loader_cls()
    ready, why = loader.available()
    if not ready:
        raise SystemExit(f"{dataset} 数据未就绪：{why}\n先跑 python scripts/prepare_datasets.py")

    samples = [s for s in loader.load() if s.bbox is not None]
    if platform:
        samples = [s for s in samples if s.platform.value == platform]
    return samples


def _screenshot_of(sample):
    """把数据集里的截图读成 `Screenshot`，供后端复用同一条编码路径。"""
    import cv2
    import numpy as np

    from perception.capture import Screenshot

    # np.fromfile 而不是 cv2.imread：后者在中文路径上会静默返回 None
    buffer = np.fromfile(sample.screenshot_path, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"读不出图片：{sample.screenshot_path}")
    height, width = image.shape[:2]
    return Screenshot(image=image, region=BBox(0, 0, width, height), engine="dataset")


def _done_ids(path: Path) -> set[str]:
    """已完成的 sample_id。**坏行跳过而不是报错**——断电时最后一行天然可能残缺。"""
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


def evaluate(
    provider: str,
    model: str = "",
    platform: str = "",
    limit: int = 0,
    space: tuple[int, int] = (1000, 1000),
    dataset: str = "screenspot",
    resume: bool = True,
) -> Path:
    """跑一档评测，结果逐条追加到 JSONL，返回结果文件路径。"""
    from llm.base import LLMBackendError
    from llm.openai_compat import OpenAICompatBackend
    from llm.providers import load_dotenv_if_present, resolve

    load_dotenv_if_present()
    config = resolve(provider, model=model or None)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{provider}-{config.model}-{platform or 'all'}".replace("/", "_")
    out = RESULT_DIR / f"{tag}.jsonl"

    samples = load_samples(platform, dataset)
    done = _done_ids(out) if resume else set()
    todo = [s for s in samples if s.sample_id not in done]
    if limit:
        todo = todo[:limit]

    print("=" * 70)
    print("ScreenSpot 定位评测")
    print("=" * 70)
    print(f"  平台      {platform or '全部'}    数据集 {dataset}")
    print(f"  模型      {config.provider.label} / {config.model}")
    print(f"  坐标空间  {space[0]}×{space[1]}  ← **接新后端时第一件要实测确认的事**")
    print(f"  样本      共 {len(samples)}，已完成 {len(done)}，本次跑 {len(todo)}")
    print(f"  结果      {out}（逐条追加，可断点续跑）")
    print()
    if not todo:
        print("  没有待跑样本。加 --no-resume 可重跑全部。")
        return out

    backend = OpenAICompatBackend(config=config, system_prompt=SYSTEM_PROMPT)
    hits = 0
    started = time.perf_counter()

    try:
        with out.open("a", encoding="utf-8") as handle:
            for index, sample in enumerate(todo, start=1):
                record = Prediction(
                    sample_id=sample.sample_id,
                    platform=sample.platform.value,
                    element_kind=str(sample.meta.get("element_kind", "")),
                    instruction=sample.instruction,
                    bbox=[
                        sample.bbox.left,
                        sample.bbox.top,
                        sample.bbox.right,
                        sample.bbox.bottom,
                    ],
                )
                step_started = time.perf_counter()
                try:
                    shot = _screenshot_of(sample)
                    intent = backend.predict_action(sample.instruction, shot)
                    x, y = intent.params.get("x"), intent.params.get("y")
                    if x is None or y is None:
                        record.error = "模型没给坐标"
                    else:
                        record.raw_xy = [x, y]
                        point = to_pixels(x, y, space, sample.resolution)
                        record.pixel_xy = [point.x, point.y]
                        record.hit = inside(point, sample.bbox)
                    record.tokens = intent.usage.total_tokens
                    record.cost_cny = round(intent.cost_cny, 6)
                except LLMBackendError as exc:
                    record.error = f"{exc.kind}: {exc}"[:200]
                except Exception as exc:  # noqa: BLE001 —— 单样本失败不该中断整档
                    record.error = f"{type(exc).__name__}: {exc}"[:200]

                record.latency_ms = round((time.perf_counter() - step_started) * 1000, 1)
                hits += record.hit
                # **每条都 flush。** 断电时丢一条和丢几百条是两码事
                handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
                handle.flush()

                if index % 10 == 0 or index == len(todo):
                    elapsed = time.perf_counter() - started
                    rate = elapsed / index
                    remain = rate * (len(todo) - index)
                    print(
                        f"  [{index}/{len(todo)}] 命中 {hits}/{index} = {hits / index:.1%}"
                        f"  {rate:.1f}s/条  预计还需 {remain / 60:.0f} 分钟"
                    )
    except KeyboardInterrupt:
        print("\n  已中断。已完成的结果都在文件里，重跑会自动续上。")
    finally:
        backend.close()

    return out


def summarize(path: Path) -> dict:
    """把一个结果文件汇总成分档统计。"""
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    valid = [r for r in rows if not r["error"]]
    stats: dict = {
        "file": path.name,
        "total": len(rows),
        "errors": len(rows) - len(valid),
        "hits": sum(r["hit"] for r in valid),
        "accuracy": (sum(r["hit"] for r in valid) / len(valid)) if valid else 0.0,
        "by_platform": {},
        "by_kind": {},
        "by_size": {},
        "tokens": sum(r.get("tokens", 0) for r in rows),
        "cost_cny": round(sum(r.get("cost_cny", 0.0) for r in rows), 6),
        "median_latency_ms": 0.0,
    }
    latencies = sorted(r["latency_ms"] for r in rows if r["latency_ms"])
    if latencies:
        stats["median_latency_ms"] = latencies[len(latencies) // 2]

    def group(key_fn) -> dict:
        buckets: dict = defaultdict(lambda: [0, 0])
        for row in valid:
            bucket = buckets[key_fn(row)]
            bucket[0] += row["hit"]
            bucket[1] += 1
        return {k: {"hit": v[0], "n": v[1], "acc": v[0] / v[1]} for k, v in sorted(buckets.items())}

    stats["by_platform"] = group(lambda r: r["platform"] or "unknown")
    stats["by_kind"] = group(lambda r: r["element_kind"] or "unknown")
    stats["by_size"] = group(
        lambda r: _bucket((r["bbox"][2] - r["bbox"][0]) * (r["bbox"][3] - r["bbox"][1]))
    )
    return stats


def print_summary(stats: dict) -> None:
    print()
    print("=" * 70)
    print(f"汇总 {stats['file']}")
    print("=" * 70)
    print(
        f"  准确率  {stats['hits']}/{stats['total'] - stats['errors']}"
        f" = **{stats['accuracy']:.1%}**    调用失败 {stats['errors']}"
    )
    print(f"  中位延迟 {stats['median_latency_ms']:.0f}ms    tokens {stats['tokens']}")
    if stats["cost_cny"]:
        print(f"  成本    {stats['cost_cny']:.4f} 元")
    for title, key in (("平台", "by_platform"), ("元素类型", "by_kind"), ("元素尺寸", "by_size")):
        if not stats[key]:
            continue
        print(f"\n  按{title}：")
        for name, v in stats[key].items():
            print(f"    {name:<12}{v['hit']:>4}/{v['n']:<5}{v['acc']:>7.1%}")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="ScreenSpot 定位准确率评测（M3）")
    parser.add_argument("--provider", default="dashscope")
    parser.add_argument("--model", default="", help="覆盖平台默认模型")
    parser.add_argument("--platform", default="", help="desktop / web / mobile，空为全部")
    parser.add_argument("--dataset", default="screenspot", help="screenspot / screenspot_v2")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条，冒烟用")
    parser.add_argument("--space", default="1000x1000", help="模型坐标空间，如 1000x1000")
    parser.add_argument("--no-resume", action="store_true", help="不跳过已完成的样本")
    parser.add_argument("--report", action="store_true", help="只汇总已有结果，不调用 API")
    args = parser.parse_args()

    if args.report:
        files = sorted(RESULT_DIR.glob("*.jsonl"))
        if not files:
            print(f"{RESULT_DIR} 下没有结果文件。")
            return 1
        for path in files:
            print_summary(summarize(path))
        return 0

    try:
        width, height = (int(v) for v in args.space.lower().split("x"))
    except ValueError:
        raise SystemExit(f"--space 格式应为 宽x高，收到 {args.space!r}") from None

    out = evaluate(
        provider=args.provider,
        model=args.model,
        platform=args.platform,
        limit=args.limit,
        space=(width, height),
        dataset=args.dataset,
        resume=not args.no_resume,
    )
    print_summary(summarize(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
