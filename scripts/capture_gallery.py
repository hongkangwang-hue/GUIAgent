"""感知效果可视化图集生成工具。

M1 交付物之一是"感知效果可视化图集，不少于 10 张标注截图"。本脚本负责
产出这些图：截屏 → 双通道识别 → 画框标注 → 存盘，并把每张图的分项统计
写进一份 JSON，方便后续统计召回率与耗时。

M1 阶段没有大模型参与，**感知效果好不好只能靠人眼判断**，这些图就是唯一
的评估依据。

用法::

    # 立即抓一张（先手动切到目标窗口）
    python scripts/capture_gallery.py --name browser

    # 倒计时 5 秒后抓，留出切窗口的时间
    python scripts/capture_gallery.py --name explorer --delay 5

    # 只用 UIA 通道（OCR 未安装或想看单通道效果时）
    python scripts/capture_gallery.py --name notepad --no-ocr
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perception.capture import ScreenCapturer  # noqa: E402
from perception.element_detector import ElementDetector  # noqa: E402
from perception.types import ElementSource  # noqa: E402
from perception.uia_tree import UIATree, get_foreground_window_title  # noqa: E402
from perception.visualizer import save_annotated  # noqa: E402

OUTPUT_DIR = Path("outputs/gallery")


def build_detector(use_ocr: bool, use_gpu: bool, budget_ms: float = 8000.0) -> ElementDetector:
    ocr_engine = None
    if use_ocr:
        try:
            from perception.ocr_engine import PaddleOCREngine

            engine = PaddleOCREngine(use_gpu=use_gpu)
            if engine.is_available():
                ocr_engine = engine
            else:
                print("[!] PaddleOCR 不可用，本次只用 UIA 通道")
        except ImportError as exc:
            print(f"[!] PaddleOCR 未安装（{exc}），本次只用 UIA 通道")
    # UIA 默认 1500ms 的遍历预算是给 Agent 循环定的——那里每步都要抓，
    # 慢一点就拖垮整个任务。**但图集采集是离线的**，一张图多花两秒无所谓，
    # 而被截断的元素树会让「按来源分项统计的召回率」低估 UIA 通道。
    #
    # 实测：客机上设置页面 1500ms 只抓到 16 个就被掐断，OCR 却有 80 个，
    # 这个对比是假的。
    return ElementDetector(ocr_engine=ocr_engine, uia_tree=UIATree(time_budget_ms=budget_ms))


def main() -> int:
    parser = argparse.ArgumentParser(description="生成感知效果可视化图集")
    parser.add_argument("--name", default="screen", help="图片名前缀，建议用场景名")
    parser.add_argument("--delay", type=float, default=0.0, help="倒计时秒数，留出切窗口时间")
    parser.add_argument("--monitor", type=int, default=1, help="显示器索引")
    parser.add_argument("--no-ocr", action="store_true", help="只用 UIA 通道")
    parser.add_argument("--no-uia", action="store_true", help="只用 OCR 通道")
    parser.add_argument("--gpu", action="store_true", help="OCR 走 GPU")
    parser.add_argument("--desktop", action="store_true", help="抓整个桌面而非前台窗口的元素树")
    parser.add_argument(
        "--uia-budget",
        type=float,
        default=8000.0,
        help=(
            "UIA 遍历时间预算（毫秒）。默认 8000，远高于 Agent 循环里用的 1500——"
            "图集采集是离线的，多花两秒无所谓，而遍历被截断会让按来源分项的"
            "召回率低估 UIA 通道"
        ),
    )
    args = parser.parse_args()

    for remaining in range(int(args.delay), 0, -1):
        print(f"\r{remaining} 秒后截图，请切换到目标窗口…", end="", flush=True)
        time.sleep(1)
    if args.delay:
        print("\r" + " " * 40 + "\r", end="")

    detector = build_detector(use_ocr=not args.no_ocr, use_gpu=args.gpu, budget_ms=args.uia_budget)

    with ScreenCapturer() as capturer:
        window_title = get_foreground_window_title()
        shot = capturer.capture(args.monitor, fresh=True)
        print(f"截图：{shot.width}×{shot.height} @ {shot.engine}，{shot.latency_ms:.2f}ms")
        if window_title:
            print(f"前台窗口：{window_title}")

        result = detector.detect(
            shot,
            use_uia=not args.no_uia,
            use_ocr=not args.no_ocr,
            foreground_only=not args.desktop,
        )

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{args.name}-{stamp}"
    image_path = OUTPUT_DIR / f"{stem}.png"
    raw_path = OUTPUT_DIR / f"{stem}.raw.png"
    detections_path = OUTPUT_DIR / f"{stem}.detections.json"

    # 原图必须单独存一份：标注真值时不能看着已经画好框的图标，
    # 否则标注会被检测结果带偏，召回率就测不出真东西了
    import cv2

    ok, buffer = cv2.imencode(".png", shot.image)
    if ok:
        raw_path.write_bytes(buffer.tobytes())

    # 检测结果与截图同时落盘。UIA 是**实时**的，无法在保存的静态图上重跑，
    # 因此召回率评测只能拿此刻的检测结果去比对，不能事后重新检测。
    detections_path.write_text(
        json.dumps(
            {
                "image": raw_path.name,
                "region": shot.region.as_tuple(),
                "window_title": window_title,
                "summary": result.summary(),
                "elements": [
                    {
                        "index": element.index,
                        "bbox": element.bbox.as_tuple(),
                        "source": element.source.value,
                        "text": element.text,
                        "control_type": element.control_type,
                        "confidence": round(element.confidence, 4),
                        "click_point": element.click_point.as_tuple(),
                    }
                    for element in result.elements
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    save_annotated(
        shot.image,
        result.elements,
        str(image_path),
        origin=(shot.region.left, shot.region.top),
        stats=result.summary(),
    )

    record = {
        "name": args.name,
        "timestamp": stamp,
        "window_title": window_title,
        "image": str(image_path),
        "raw_image": str(raw_path),
        "detections": str(detections_path),
        "engine": shot.engine,
        "capture_latency_ms": round(shot.latency_ms, 2),
        **result.summary(),
        "by_source": {
            "uia": len(result.by_source(ElementSource.UIA)),
            "ocr": len(result.by_source(ElementSource.OCR)),
        },
    }
    index_path = OUTPUT_DIR / "index.jsonl"
    with open(index_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print()
    print(f"  UIA 原始     {result.uia_count:4d} 个  ({result.uia_ms:.0f}ms)")
    # 截断原因决定该怎么调：time 就加预算，count 就提上限，depth 就加深度。
    # 光看到「已截断」三个字是没法判断的，而遍历不完整会让按来源分项的
    # 召回率低估 UIA 通道。
    _uia = result.uia_stats or {}
    if _uia.get("truncated_by", "none") != "none":
        print(
            f"             ↑ 被 {_uia['truncated_by']} 截断"
            f"（访问 {_uia.get('visited', '?')} 个节点，"
            f"最深 {_uia.get('max_depth_reached', '?')} 层）"
            f" —— 用 --uia-budget 加预算"
        )
    print(f"  OCR 原始     {result.ocr_count:4d} 个  ({result.ocr_ms:.0f}ms)")
    print(f"  融合去重后   {result.fused_count:4d} 个  (丢弃 {result.dropped_by_dedupe} 个重复)")
    print(f"  识别总耗时   {result.total_ms:.0f}ms")
    print()
    print(f"已保存 标注图 {image_path}")
    print(f"       原图   {raw_path}（标注真值用这张）")
    print(f"       检测   {detections_path}")
    print(f"已追加 {index_path}")

    existing = len(list(OUTPUT_DIR.glob("*.raw.png")))
    print(
        f"图集当前共 {existing} 张"
        + ("（交付要求不少于 10 张，召回率评测需 20 张）" if existing < 20 else " ✓")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
