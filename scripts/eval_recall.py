"""召回率评测 —— M1 验收标准 4。

> 在 20 张测试截图上**报告**双通道融合后的 UI 元素召回率实测值，
> 按来源分项统计（不设硬性门槛）。

## 判据用"点击点命中"，不用 IoU

传统检测任务用 IoU ≥ 0.5 判命中。但对 GUI Agent 来说，**框重合多少不
重要，点下去能不能点中才重要**：

- OCR 把按钮上的文字框成一个小框，与按钮真值框的 IoU 可能只有 0.2，
  但它的中心点稳稳落在按钮里 —— 点下去完全正确
- 反过来，一个横跨两个按钮的大框可能 IoU 达标，中心点却落在两者之间
  的缝隙上 —— 点下去什么都没点到

因此主判据是 **detected.click_point ∈ ground_truth.bbox**。IoU 同时也算
并报告，用作对照，让读者知道两种口径的差距有多大。

## 为什么不能事后重新检测

**UIA 是实时的**，无法在保存的静态图上重跑。因此本脚本比对的是
`capture_gallery.py` 截图当时一并落盘的 `*.detections.json`，
而不是现场重新检测。这也意味着截图与检测结果必须成对保管，
删了任何一半这张图就作废了。

## 用法

    python scripts/eval_recall.py                       # 评测 outputs/gallery 下全部
    python scripts/eval_recall.py --dir outputs/gallery --iou 0.5
    python scripts/eval_recall.py --json > recall.json  # 机器可读
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perception.types import BBox, Point  # noqa: E402

DEFAULT_DIR = Path("outputs/gallery")
SOURCES = ("uia", "ocr")


@dataclass
class Match:
    """一个真值元素的匹配结果。"""

    label: str
    element_type: str
    hit_by_click: bool = False
    hit_by_iou: bool = False
    #: 命中它的检测元素来自哪个通道（点击点判据下）
    hit_source: str = ""
    best_iou: float = 0.0
    matched_text: str = ""


@dataclass
class ImageReport:
    name: str
    total_gt: int = 0
    matches: list[Match] = field(default_factory=list)
    num_detections: int = 0
    detections_by_source: dict = field(default_factory=dict)

    def recall(self, criterion: str = "click", source: str | None = None) -> float:
        if not self.total_gt:
            return 0.0
        hits = sum(1 for m in self.matches if self._is_hit(m, criterion, source))
        return hits / self.total_gt

    @staticmethod
    def _is_hit(match: Match, criterion: str, source: str | None) -> bool:
        hit = match.hit_by_click if criterion == "click" else match.hit_by_iou
        if source is None:
            return hit
        return hit and match.hit_source == source


def evaluate_image(gt: dict, detections: dict, iou_threshold: float) -> ImageReport:
    """把一张图的真值与检测结果比对。

    真值坐标是**图像相对坐标**（标注工具在图上画的），检测结果是**屏幕
    绝对坐标**（UIA 天然如此，OCR 已在 element_detector 里平移过）。
    因此比对前必须减掉截图区域的原点 —— 截全屏时两者恰好重合，这个
    差别只在截区域或副显示器时才暴露。
    """
    region = detections.get("region", [0, 0, 0, 0])
    off_x, off_y = region[0], region[1]

    detected = []
    for item in detections["elements"]:
        left, top, right, bottom = item["bbox"]
        detected.append(
            {
                "bbox": BBox(left - off_x, top - off_y, right - off_x, bottom - off_y),
                "click_point": Point(
                    item["click_point"][0] - off_x, item["click_point"][1] - off_y
                ),
                "source": item["source"],
                "text": item.get("text", ""),
            }
        )

    by_source: dict[str, int] = defaultdict(int)
    for item in detected:
        by_source[item["source"]] += 1

    report = ImageReport(
        name=gt.get("image", "?"),
        total_gt=len(gt["elements"]),
        num_detections=len(detected),
        detections_by_source=dict(by_source),
    )

    for element in gt["elements"]:
        truth_box = BBox(*element["bbox"])
        match = Match(label=element.get("label", ""), element_type=element.get("type", "other"))

        for item in detected:
            # 主判据：这个检测元素的点击点是否落在真值框内
            if not match.hit_by_click and truth_box.contains(item["click_point"]):
                match.hit_by_click = True
                match.hit_source = item["source"]
                match.matched_text = item["text"]

            iou = truth_box.iou(item["bbox"])
            if iou > match.best_iou:
                match.best_iou = iou
            if iou >= iou_threshold:
                match.hit_by_iou = True

        report.matches.append(match)

    return report


def aggregate(reports: list[ImageReport]) -> dict:
    total_gt = sum(r.total_gt for r in reports)
    if not total_gt:
        return {"images": len(reports), "total_gt": 0}

    def overall(criterion: str, source: str | None = None) -> float:
        hits = sum(
            sum(1 for m in r.matches if ImageReport._is_hit(m, criterion, source)) for r in reports
        )
        return round(hits / total_gt, 4)

    by_type: dict[str, dict] = defaultdict(lambda: {"total": 0, "hit": 0})
    for report in reports:
        for match in report.matches:
            entry = by_type[match.element_type]
            entry["total"] += 1
            entry["hit"] += int(match.hit_by_click)

    return {
        "images": len(reports),
        "total_gt": total_gt,
        "total_detections": sum(r.num_detections for r in reports),
        "recall_click": overall("click"),
        "recall_iou": overall("iou"),
        "recall_click_by_source": {s: overall("click", s) for s in SOURCES},
        "recall_by_type": {
            t: {**v, "recall": round(v["hit"] / v["total"], 4)} for t, v in sorted(by_type.items())
        },
    }


def render(reports: list[ImageReport], summary: dict, iou_threshold: float) -> None:
    print("=" * 74)
    print("UI 元素召回率评测（口径：任务相关元素）")
    print("=" * 74)
    print(f"{'截图':<34}{'真值':>5}{'检测':>6}{'点击命中':>10}{'IoU命中':>10}")
    print("-" * 74)
    for report in reports:
        print(
            f"{report.name[:33]:<34}{report.total_gt:>5}{report.num_detections:>6}"
            f"{report.recall('click'):>9.1%}{report.recall('iou'):>10.1%}"
        )
    print("-" * 74)
    print(
        f"{'合计':<34}{summary['total_gt']:>5}{summary['total_detections']:>6}"
        f"{summary['recall_click']:>9.1%}{summary['recall_iou']:>10.1%}"
    )
    print()

    print("按来源分项（点击点判据，谁先命中算谁的）：")
    for source, value in summary["recall_click_by_source"].items():
        print(f"  {source:<8}{value:>7.1%}")
    print()

    print("按元素类型：")
    for element_type, stats in summary["recall_by_type"].items():
        print(f"  {element_type:<12}{stats['hit']:>3}/{stats['total']:<4}{stats['recall']:>8.1%}")
    print()

    gap = summary["recall_click"] - summary["recall_iou"]
    print(
        f"两种判据相差 {gap:+.1%}。"
        + (
            "点击点判据更宽松，说明有一批检测框与真值重合度不高、但点击位置正确"
            "——对 GUI Agent 而言这些算命中。"
            if gap > 0.02
            else "两者接近，检测框与真值框基本吻合。"
        )
    )
    print(f"（IoU 阈值 {iou_threshold}）")
    print()

    missed = [
        (r.name, m) for r in reports for m in r.matches if not m.hit_by_click
    ]
    if missed:
        print(f"漏检 {len(missed)} 个，前 15 个：")
        for name, match in missed[:15]:
            print(f"  {name[:28]:<30}{match.element_type:<12}{match.label[:24]:<26}"
                  f"best_iou={match.best_iou:.2f}")
        print()
        print("⚠ 漏检清单是 M4「提升识别准确率」那条任务的输入。")
        print("  M4 明确要求改动必须有失败案例支撑，禁止凭直觉调参数——这份清单就是那些案例。")


def main() -> int:
    parser = argparse.ArgumentParser(description="UI 元素召回率评测")
    parser.add_argument("--dir", default=str(DEFAULT_DIR), help="图集目录")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU 判据阈值（对照用）")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args()

    directory = Path(args.dir)
    gt_files = sorted(directory.glob("*.gt.json"))
    if not gt_files:
        print(f"{directory} 下没有 *.gt.json。")
        print("先用 capture_gallery.py 截图，再用 annotate.py 标注真值。")
        return 1

    reports = []
    skipped = []
    for gt_path in gt_files:
        detections_path = gt_path.with_name(gt_path.name.replace(".gt.json", ".detections.json"))
        if not detections_path.exists():
            skipped.append(f"{gt_path.name}：缺少对应的 .detections.json")
            continue
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        detections = json.loads(detections_path.read_text(encoding="utf-8"))
        reports.append(evaluate_image(gt, detections, args.iou))

    if not reports:
        print("没有可评测的图片对：")
        for reason in skipped:
            print(f"  {reason}")
        return 1

    summary = aggregate(reports)

    if args.json:
        print(
            json.dumps(
                {
                    "summary": summary,
                    "iou_threshold": args.iou,
                    "per_image": [
                        {
                            "name": r.name,
                            "total_gt": r.total_gt,
                            "num_detections": r.num_detections,
                            "recall_click": round(r.recall("click"), 4),
                            "recall_iou": round(r.recall("iou"), 4),
                            "missed": [
                                {"label": m.label, "type": m.element_type, "best_iou": round(m.best_iou, 3)}
                                for m in r.matches
                                if not m.hit_by_click
                            ],
                        }
                        for r in reports
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        render(reports, summary, args.iou)
        if skipped:
            print()
            print(f"跳过 {len(skipped)} 张：")
            for reason in skipped:
                print(f"  {reason}")
        if len(reports) < 20:
            print()
            print(f"⚠ 当前只评了 {len(reports)} 张，验收要求 20 张。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
