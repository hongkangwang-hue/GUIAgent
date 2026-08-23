"""生成 OCR 对照实验的合成测试集 —— 已知文本 + 已知位置。

M1 交付物《OCR 双引擎对照实验小结》要求记录**中文准确率、英文准确率、
平均耗时**。耗时用真实截图测就行，但**准确率需要逐字真值**，而我们没有。

## 为什么不用真实截图的标注真值

`annotate.py` 标的是「元素框 + 元素名」，用途是召回率。它的标签是人凭
印象打的（很多直接回车留空），不是屏幕上文字的逐字转录。拿它当 OCR
真值会把「人打的标签和 OCR 读出来的字不一样」误判成 OCR 错误。

## 也不能拿一个引擎的输出当另一个的真值

那只能证明两者像不像，不能证明谁更准。`ocr_benchmark.py` 的文档里已经
写死了这条原则。

## 所以合成

把已知字符串渲染到 GUI 风格的背景上，真值就是渲染时用的那个字符串，
位置就是渲染时用的那个框——**逐字精确，零标注成本，完全可复现**。

代价是合成图比真实截图「干净」：没有半透明叠加、没有抗锯齿差异、
没有复杂背景。因此本测试集测的是**两个引擎在清晰文本上的字符识别能力
上限**，不能外推到真实截图的识别率。真实截图那一半用 ScreenSpot-v2 的
桌面子集跑，只报吞吐与延迟。两组数据分开呈现，各自说明各自的边界。

## 用法

    python scripts/make_ocr_testset.py                 # 默认 6 张，输出到 data/ocr-testset
    python scripts/make_ocr_testset.py --count 10
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT_DIR = Path("data/ocr-testset")

#: 中文短语。取自真实桌面界面的常见文案——按钮、菜单项、设置项，
#: 而不是随机汉字：OCR 的语言模型会用上下文纠错，用无意义字符串测
#: 会低估它在真实界面上的表现。
CHINESE = [
    "文件",
    "编辑",
    "查看",
    "帮助",
    "设置",
    "保存",
    "另存为",
    "打开文件",
    "新建标签页",
    "关闭窗口",
    "复制",
    "粘贴",
    "剪切",
    "全选",
    "撤销",
    "蓝牙和其他设备",
    "网络和 Internet",
    "个性化",
    "隐私和安全性",
    "系统更新",
    "存储空间",
    "显示器分辨率",
    "缩放和布局",
    "默认应用",
    "确定",
    "取消",
    "应用",
    "下一步",
    "上一步",
    "完成",
]

#: 英文短语。同理取自真实界面。
ENGLISH = [
    "File",
    "Edit",
    "View",
    "Help",
    "Settings",
    "Save",
    "Save As",
    "Open File",
    "New Tab",
    "Close Window",
    "Copy",
    "Paste",
    "Cut",
    "Select All",
    "Undo",
    "Bluetooth & devices",
    "Network & internet",
    "Personalization",
    "Privacy & security",
    "Windows Update",
    "Storage",
    "Display resolution",
    "Scale & layout",
    "Default apps",
    "OK",
    "Cancel",
    "Apply",
    "Next",
    "Back",
    "Finish",
]

#: 中英混排。真实界面里很常见，而两个引擎对混排的处理差异往往最大。
MIXED = [
    "Wi-Fi 设置",
    "PDF 导出",
    "USB 设备",
    "CPU 占用率",
    "IP 地址",
    "OK 确定",
    "Windows 更新",
    "Microsoft 账户",
    "OneDrive 同步",
]


def _font(size: int):
    """找一个能渲染中文的字体。找不到就报错退出，不静默降级。

    降级成默认字体会渲染出一堆方框，而那看起来像「OCR 认不出中文」——
    实际是图里根本没有中文。这类假失败必须挡在生成阶段。
    """
    from PIL import ImageFont

    candidates = [
        "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
        "C:/Windows/Fonts/simhei.ttf",  # 黑体
        "C:/Windows/Fonts/simsun.ttc",  # 宋体
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    raise SystemExit(
        "找不到可渲染中文的字体。合成图里的中文会变成方框，"
        "那会被误读成「OCR 认不出中文」。请安装中文字体后重试。"
    )


def make_image(index: int, rng: random.Random, width: int = 1280, height: int = 800):
    """渲染一张 GUI 风格的图，返回 (PIL 图, 真值元素列表)。"""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (width, height), (243, 243, 243))
    draw = ImageDraw.Draw(image)

    # 左侧导航面板 + 右侧内容区，模仿 Windows 11 设置的版式
    draw.rectangle([0, 0, 300, height], fill=(249, 249, 249))
    draw.rectangle([300, 0, width, 64], fill=(255, 255, 255))
    draw.line([300, 0, 300, height], fill=(224, 224, 224), width=1)
    draw.line([300, 64, width, 64], fill=(224, 224, 224), width=1)

    pool = CHINESE + ENGLISH + MIXED
    phrases = rng.sample(pool, k=min(24, len(pool)))

    elements = []
    font_body = _font(18)
    font_title = _font(24)

    # 标题
    title = phrases.pop()
    draw.text((330, 20), title, fill=(20, 20, 20), font=font_title)
    box = draw.textbbox((330, 20), title, font=font_title)
    elements.append({"label": title, "type": "text", "bbox": list(box)})

    # 左侧导航项
    y = 90
    for _ in range(8):
        if not phrases:
            break
        text = phrases.pop()
        draw.rectangle([16, y - 8, 284, y + 30], fill=(255, 255, 255))
        draw.text((48, y), text, fill=(30, 30, 30), font=font_body)
        box = draw.textbbox((48, y), text, font=font_body)
        elements.append({"label": text, "type": "menu_item", "bbox": list(box)})
        y += 52

    # 右侧内容卡片
    y = 110
    for _ in range(6):
        if not phrases:
            break
        text = phrases.pop()
        draw.rectangle([330, y - 12, width - 40, y + 44], fill=(255, 255, 255))
        draw.text((360, y + 6), text, fill=(30, 30, 30), font=font_body)
        box = draw.textbbox((360, y + 6), text, font=font_body)
        elements.append({"label": text, "type": "button", "bbox": list(box)})
        y += 76

    return image, elements


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="生成 OCR 对照实验的合成测试集")
    parser.add_argument("--count", type=int, default=6, help="生成几张")
    parser.add_argument("--seed", type=int, default=20260823, help="随机种子，固定以便复现")
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    total_elements = 0
    for index in range(1, args.count + 1):
        image, elements = make_image(index, rng)
        stem = f"synth-{index:02d}"
        image_path = out_dir / f"{stem}.raw.png"
        image.save(image_path)

        gt = {
            "image": str(image_path),
            "resolution": list(image.size),
            "source": "synthetic",
            "note": "合成测试集，真值即渲染时使用的字符串，逐字精确",
            "elements": elements,
        }
        (out_dir / f"{stem}.gt.json").write_text(
            json.dumps(gt, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        total_elements += len(elements)
        print(f"  {image_path}  {len(elements)} 条真值")

    print(f"\n共 {args.count} 张，{total_elements} 条逐字真值 → {out_dir}")
    print("接着跑：python scripts/ocr_benchmark.py --dir " + str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
