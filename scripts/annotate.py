"""真值标注工具 —— 为召回率评测准备 ground truth。

M1 验收标准 4 要求"在 20 张测试截图上报告双通道融合后的 UI 元素召回率
实测值，按来源分项统计"。召回率需要真值，真值需要人工标。

## 两个方法学上的决定，直接影响结论是否可信

**① 不用检测结果预填真值。**

看起来很省事：把 UIA + OCR 检测到的元素列出来，人工勾掉错的、补上漏的。
但这会**系统性地低估漏检**——你不会去标一个检测器完全没提到、你自己
也没注意到的元素，而那恰恰是召回率要衡量的东西。因此本工具在**未标注
的原图**上手工画框，全程不显示任何检测结果。

`capture_gallery.py` 存的 `*.raw.png` 就是给这一步用的，标注时不要打开
带框的那张。

**② 只标"任务相关元素"，不标全部文字。**

全屏 2560×1600 的截图上有几百个文字块，全标一遍 20 张要好几天，且大部分
文字（正文、日志、状态栏）Agent 根本不需要点。因此只标**一个 GUI 任务
会去交互的东西**：按钮、菜单项、输入框、可点击的列表项、标签页、复选框。
每张 10-20 个。

这样得到的是**任务相关元素召回率**，比"全文字召回率"更贴近系统实际要求，
也是唯一在 M1 预算内能做完的做法。报告里必须写明这个口径，不能笼统称
"召回率 X%"。

## 用法

    python scripts/annotate.py outputs/gallery/browser-20260818-114316.raw.png

操作：
    鼠标拖拽画框 → 回车/空格 确认这一个 → 继续画下一个 → ESC 结束
    画完后在终端逐个输入标签（直接回车用默认标签）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#: 元素类型。标注时输入编号即可，省得敲字
ELEMENT_TYPES = {
    "1": "button",
    "2": "menu_item",
    "3": "input",
    "4": "list_item",
    "5": "tab",
    "6": "checkbox",
    "7": "link",
    "8": "icon",
    "9": "other",
}


def annotate(image_path: Path, existing: dict | None = None) -> dict:
    import cv2

    image = cv2.imdecode(
        __import__("numpy").fromfile(str(image_path), dtype="uint8"), cv2.IMREAD_COLOR
    )
    if image is None:
        raise SystemExit(f"读不出图片：{image_path}")

    height, width = image.shape[:2]
    print(f"图片 {width}×{height}")
    print()
    print("  拖拽画框 → 回车/空格 确认 → 继续画下一个 → ESC 结束")
    print("  只标任务相关元素（按钮 / 菜单项 / 输入框 / 列表项 / 标签页 / 复选框）")
    print("  不要标正文、日志、状态栏这类不会被点击的文字")
    print()

    window = "annotate (ESC 结束)"
    # showCrosshair=True 让边缘对齐更容易；fromCenter=False 是从左上角拖
    boxes = cv2.selectROIs(window, image, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    if len(boxes) == 0:
        print("没有画任何框，退出")
        return existing or {}

    print(f"\n共 {len(boxes)} 个框。逐个输入标签：")
    print("  格式：<文字>  或  <文字>|<类型编号>")
    print("  类型：" + "  ".join(f"{k}={v}" for k, v in ELEMENT_TYPES.items()))
    print()

    elements = []
    for i, (x, y, w, h) in enumerate(boxes, start=1):
        raw = input(f"  [{i}/{len(boxes)}] ({x},{y}) {w}×{h} → ").strip()
        if "|" in raw:
            label, type_key = (part.strip() for part in raw.split("|", 1))
        else:
            label, type_key = raw, "9"
        elements.append(
            {
                "label": label or f"element_{i}",
                "type": ELEMENT_TYPES.get(type_key, "other"),
                "bbox": [int(x), int(y), int(x + w), int(y + h)],
            }
        )

    return {
        "image": image_path.name,
        "image_size": [width, height],
        "annotation_scope": "task_relevant",  # 见模块文档②
        "elements": elements,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="标注真值，供召回率评测使用")
    parser.add_argument("image", help="要标注的原图（*.raw.png，不要用带框的那张）")
    parser.add_argument("--append", action="store_true", help="追加到已有标注而非覆盖")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"文件不存在：{image_path}")
    if ".raw." not in image_path.name:
        print(f"⚠ {image_path.name} 看起来不是原图。")
        print("  标注必须在未画框的原图（*.raw.png）上做，否则真值会被检测结果带偏。")
        if input("  仍要继续？(y/N) ").strip().lower() != "y":
            return 1

    gt_path = image_path.with_name(image_path.name.replace(".raw.png", "") + ".gt.json")
    existing = None
    if gt_path.exists():
        existing = json.loads(gt_path.read_text(encoding="utf-8"))
        print(f"已有标注 {len(existing['elements'])} 个" + ("，本次追加" if args.append else "，本次覆盖"))

    result = annotate(image_path, existing)
    if not result:
        return 0

    if args.append and existing:
        result["elements"] = existing["elements"] + result["elements"]

    gt_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存 {gt_path}（共 {len(result['elements'])} 个真值元素）")

    done = len(list(image_path.parent.glob("*.gt.json")))
    print(f"已标注 {done} 张" + ("（召回率评测需 20 张）" if done < 20 else " ✓"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
