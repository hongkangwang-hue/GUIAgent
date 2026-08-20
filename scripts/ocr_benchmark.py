"""OCR 双引擎对照实验 —— M1 交付物《OCR 双引擎对照实验小结》。

对照 PaddleOCR 与 EasyOCR 在**同一批图、同一套预处理**下的表现。交付要求
记录中文准确率、英文准确率、平均耗时。

## 三组预处理配置，其中一组必须是空跑

- `passthrough`：什么都不做。**这是对照组，不能省**——没有它就无法证明
  预处理到底是帮忙还是帮倒忙
- `default`：灰度 + CLAHE（温和增强，**不超分**）
- `upscale2x`：再加 2× 超分。M1 之前的默认值，实测在整屏截图上净亏损
  （+51s 换 +1 框），降级为对照臂以便复现该结论
- `aggressive`：再加二值化与去噪

现代 OCR 的检测识别网络是在自然图像上训练的，**喂二值化图往往更差**：
笔画被阈值切断、抗锯齿信息丢失。"二值化提升 OCR"的经验来自 Tesseract
时代的传统算法。这个实验就是要把这件事在本项目的数据上测出来，而不是
沿用传闻。

## 准确率怎么算

耗时与产出量是客观的，直接测。**准确率需要真值**：

- 有 `*.gt.json`（`annotate.py` 标的）时，按真值文本算字符级准确率
- 没有真值时只报告产出量与耗时，准确率留空 —— 不猜、不用一个引擎的
  结果当另一个的真值（那只会证明两者像不像，不是谁更准）

中英文分开统计：中文靠字符集判断（CJK 区间），中英混排的条目两边都计入。

## 用法

    python scripts/ocr_benchmark.py                        # 跑 outputs/gallery 下全部原图
    python scripts/ocr_benchmark.py --images a.png b.png
    python scripts/ocr_benchmark.py --configs passthrough default
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perception.preprocess import AGGRESSIVE, DEFAULT, PASSTHROUGH, UPSCALE2X  # noqa: E402
from perception.types import BBox  # noqa: E402

CONFIGS = {
    "passthrough": PASSTHROUGH,
    "default": DEFAULT,
    "upscale2x": UPSCALE2X,
    "aggressive": AGGRESSIVE,
}
DEFAULT_DIR = Path("outputs/gallery")


def is_cjk(char: str) -> bool:
    """字符是否属于中日韩统一表意文字区。"""
    code = ord(char)
    return (
        0x4E00 <= code <= 0x9FFF  # 基本区
        or 0x3400 <= code <= 0x4DBF  # 扩展 A
        or 0xF900 <= code <= 0xFAFF  # 兼容表意文字
    )


def classify(text: str) -> str:
    """把一条文本判为 chinese / english / mixed。"""
    has_cjk = any(is_cjk(c) for c in text)
    has_latin = any(c.isascii() and c.isalnum() for c in text)
    if has_cjk and has_latin:
        return "mixed"
    return "chinese" if has_cjk else "english"


def char_accuracy(predicted: str, truth: str) -> float:
    """字符级准确率 = 1 - 归一化编辑距离。

    用编辑距离而不是精确匹配：OCR 认错一个字和整条认错完全不是一回事，
    精确匹配会把"识别了 19/20 个字"和"什么都没识别出来"记成同样的失败。
    """
    if not truth:
        return 1.0 if not predicted else 0.0
    # 标准 Levenshtein，逐行滚动数组
    previous = list(range(len(truth) + 1))
    for i, pc in enumerate(predicted, start=1):
        current = [i]
        for j, tc in enumerate(truth, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (pc != tc)))
        previous = current
    return max(0.0, 1.0 - previous[-1] / len(truth))


@dataclass
class EngineRun:
    engine: str
    config: str
    image: str
    elapsed_ms: float
    num_results: int
    mean_confidence: float
    texts: list[str] = field(default_factory=list)
    #: 有真值时才有
    accuracy_by_lang: dict = field(default_factory=dict)
    matched_gt: int = 0
    total_gt: int = 0


def score_against_gt(results, gt: dict) -> tuple[dict, int, int]:
    """按真值框位置配对，算各语种的字符准确率。

    配对方式：真值框与 OCR 框有重叠即视为同一条。**不按顺序配对**——
    OCR 的输出顺序不稳定，按顺序配对会把整体错位算成全错。
    """
    buckets: dict[str, list[float]] = {"chinese": [], "english": [], "mixed": []}
    matched = 0

    for element in gt["elements"]:
        truth_text = (element.get("label") or "").strip()
        if not truth_text:
            continue
        truth_box = BBox(*element["bbox"])

        best_text, best_overlap = "", 0
        for item in results:
            overlap = truth_box.intersection_area(item.bbox)
            if overlap > best_overlap:
                best_overlap, best_text = overlap, item.text

        if best_overlap > 0:
            matched += 1
        buckets[classify(truth_text)].append(char_accuracy(best_text.strip(), truth_text))

    accuracy = {
        lang: round(sum(scores) / len(scores), 4) for lang, scores in buckets.items() if scores
    }
    total = sum(len(v) for v in buckets.values())
    return accuracy, matched, total


def run_one(engine, config_name: str, image_path: Path, image, gt: dict | None) -> EngineRun:
    results = engine.recognize(image)
    benchmark = engine.last_benchmark

    run = EngineRun(
        engine=engine.name,
        config=config_name,
        image=image_path.name,
        elapsed_ms=benchmark.elapsed_ms,
        num_results=benchmark.num_results,
        mean_confidence=benchmark.mean_confidence,
        texts=[r.text for r in results],
    )
    if gt:
        run.accuracy_by_lang, run.matched_gt, run.total_gt = score_against_gt(results, gt)
    return run


def render(runs: list[EngineRun]) -> None:
    print("=" * 92)
    print("OCR 双引擎对照实验")
    print("=" * 92)

    grouped: dict[tuple[str, str], list[EngineRun]] = {}
    for run in runs:
        grouped.setdefault((run.engine, run.config), []).append(run)

    header = f"{'引擎':<12}{'预处理':<14}{'图数':>5}{'平均条数':>10}{'平均耗时':>11}{'中文':>9}{'英文':>9}{'混排':>9}"
    print(header)
    print("-" * 92)

    for (engine, config), group in sorted(grouped.items()):
        n = len(group)
        mean_ms = sum(r.elapsed_ms for r in group) / n
        mean_count = sum(r.num_results for r in group) / n

        def lang_mean(key: str, rows: list[EngineRun] = group) -> str:
            values = [r.accuracy_by_lang[key] for r in rows if key in r.accuracy_by_lang]
            return f"{sum(values) / len(values):.1%}" if values else "—"

        print(
            f"{engine:<12}{config:<14}{n:>5}{mean_count:>10.1f}{mean_ms:>10.0f}ms"
            f"{lang_mean('chinese'):>9}{lang_mean('english'):>9}{lang_mean('mixed'):>9}"
        )

    print("-" * 92)
    print()

    has_gt = any(r.total_gt for r in runs)
    if not has_gt:
        print("⚠ 没有找到 *.gt.json，本次只有耗时与产出量，没有准确率。")
        print("  先用 annotate.py 标注真值，准确率才有意义——")
        print("  用一个引擎的输出当另一个的真值只能证明两者像不像，不能证明谁更准。")
        print()

    # 空跑组是对照基准，把其余组和它比
    passthrough = [r for r in runs if r.config == "passthrough"]
    if passthrough and len(grouped) > 1:
        base = {}
        for run in passthrough:
            base.setdefault(run.engine, []).append(run)
        print("相对空跑组的变化（预处理到底有没有用）：")
        for (engine, config), group in sorted(grouped.items()):
            if config == "passthrough" or engine not in base:
                continue
            base_ms = sum(r.elapsed_ms for r in base[engine]) / len(base[engine])
            base_count = sum(r.num_results for r in base[engine]) / len(base[engine])
            this_ms = sum(r.elapsed_ms for r in group) / len(group)
            this_count = sum(r.num_results for r in group) / len(group)
            print(
                f"  {engine}/{config:<12} 条数 {this_count - base_count:+.1f}"
                f"   耗时 {this_ms - base_ms:+.0f}ms（{this_ms / base_ms:.1f}×）"
            )
        print()
        print("  条数变多不等于变好——可能是把噪点认成了文字。要结合准确率一起看。")


def main() -> int:
    parser = argparse.ArgumentParser(description="OCR 双引擎对照实验")
    parser.add_argument("--dir", default=str(DEFAULT_DIR))
    parser.add_argument("--images", nargs="*", help="指定图片，默认用 --dir 下全部 *.raw.png")
    parser.add_argument("--configs", nargs="*", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--out", default="outputs/ocr_benchmark.json")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    import numpy as np

    from perception.ocr_engine import EasyOCREngine, PaddleOCREngine

    if args.images:
        images = [Path(p) for p in args.images]
    else:
        images = sorted(Path(args.dir).glob("*.raw.png"))
    if not images:
        print(f"{args.dir} 下没有 *.raw.png。先用 capture_gallery.py 截图。")
        return 1

    print(f"图片 {len(images)} 张，预处理配置 {args.configs}")

    runs: list[EngineRun] = []
    for config_name in args.configs:
        config = CONFIGS[config_name]
        for engine_cls in (PaddleOCREngine, EasyOCREngine):
            engine = engine_cls(config=config, use_gpu=args.gpu)
            if not engine.is_available():
                print(f"  跳过 {engine.name}：不可用")
                continue
            for image_path in images:
                import cv2

                image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    continue
                gt_path = image_path.with_name(image_path.name.replace(".raw.png", "") + ".gt.json")
                gt = json.loads(gt_path.read_text(encoding="utf-8")) if gt_path.exists() else None
                runs.append(run_one(engine, config_name, image_path, image, gt))
                print(f"  {engine.name}/{config_name} {image_path.name} 完成")

    if not runs:
        print("没有可用的 OCR 引擎。")
        return 1

    render(runs)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps([r.__dict__ for r in runs], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"原始数据已存 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
