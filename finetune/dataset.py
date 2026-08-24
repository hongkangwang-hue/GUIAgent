"""把统一 schema 的样本转成 grounding 训练格式 —— M3 任务 1。

## 训练什么

一条样本是「截图 + 元素描述 → 坐标点」。模型学的是**在给定界面里
找到被描述的元素**，不是学怎么规划任务。目标单一，指标单一
（ScreenSpot 命中率），结论才干净。

## 坐标写成什么

**归一化到 [0,1000) 的整数**，与 `SessionConfig.coordinate_space` 一致。

这不是随便定的：`qwen3-vl` 的原生输出就是这个空间，实测过——同一张
2560×1600 截图分别告诉模型画布是 1024×768 和 1024×640，y 值都恒在 968
上下不随声明变化，968/1000 正是任务栏在 1600 高屏幕上的位置。

**训练时用与推理时不同的坐标空间，是最容易犯又最难查的错**：loss 会
正常下降，评测分数却莫名其妙地低，因为模型学的坐标系和评测换算用的
不是一个。所以这个值在 `COORD_SPACE` 里写死并被单测钉住。

## 为什么写 JSONL 而不是直接喂 Dataset

三个理由：

1. **可检查。** 转换错了要能一眼看出来——JSONL 可以直接 `head`。
2. **可复现。** 训练脚本读的是文件，不是内存里的对象，重跑必然一致。
3. **切断依赖。** 训练在租来的 GPU 上跑，那台机器不需要装 `data/` 那一整套
   （opencv、paddleocr 等），只要能读 JSONL 和图片。

## 冻结划分必须先核指纹

`FrozenSplit.read()` 会按内容重算指纹并与文件里记的比对，不符即拒绝载入。
**条数一样但内容不同的划分是完全可能出现的**（换个种子就是），
只看条数的核对等于没核。

## 用法

    python -m finetune.dataset                    # 用默认冻结划分
    python -m finetune.dataset --out finetune/data
    python -m finetune.dataset --check            # 只核对，不生成
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

#: 冻结划分。M3 开工第一件事是核对这个文件的指纹。
SPLIT_FILE = Path("docs/m3-prereq/split-screenagent-desktop.json")

#: 输出目录。
OUT_DIR = Path("finetune/data")

#: **训练坐标空间，与推理时必须一致。** 见模块 docstring。
COORD_SPACE = (1000, 1000)

#: 训练用的提示词。**与 `eval/grounding.py` 的评测提示词保持同构**——
#: 训练时说「找到并返回坐标」，评测时也这么说，模型才不用跨越
#: 提示词分布的鸿沟。措辞不必逐字相同，但任务表述必须是同一件事。
INSTRUCTION_TEMPLATE = "在这张界面截图中找到：{description}。返回它的中心坐标。"


def normalize_point(x: int, y: int, size: tuple[int, int]) -> tuple[int, int]:
    """图像像素 → 归一化 [0,1000) 整数。

    **不做四舍五入到 1000。** 归一化空间是左闭右开的 [0,1000)，
    坐标恰好落在图像最右一列时 `x/width*1000` 会等于 1000，越界。
    夹到 999 而不是让它溢出——这类边界值在几千条样本里必然出现。
    """
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError(f"分辨率非法：{size}")
    nx = min(int(x / width * COORD_SPACE[0]), COORD_SPACE[0] - 1)
    ny = min(int(y / height * COORD_SPACE[1]), COORD_SPACE[1] - 1)
    return max(nx, 0), max(ny, 0)


def describe(sample) -> str:
    """从样本里取出「元素描述」。

    优先级：指令文本 > 元素类型 > 空。ScreenAgent 的指令本身就是
    「点击左上角的文件菜单」这类自然语言，去掉动作词后正是元素描述。

    **描述为空的样本要丢掉而不是填占位符。** 「找到：」后面什么都没有
    的训练样本会教模型在缺信息时也硬猜坐标——那正是 M2 实测里最麻烦的
    行为之一。
    """
    text = (sample.instruction or "").strip()
    if text:
        return text
    kind = str(sample.meta.get("element_kind", "")).strip()
    return kind


def build_record(sample) -> dict | None:
    """一条训练样本。缺关键字段时返回 None（调用方计数丢弃原因）。"""
    description = describe(sample)
    if not description:
        return None

    point = sample.point
    if point is None and sample.bbox is not None:
        point = sample.bbox.center
    if point is None:
        return None

    if not sample.resolution or sample.resolution[0] <= 0:
        return None

    nx, ny = normalize_point(point.x, point.y, sample.resolution)
    record = {
        "sample_id": sample.sample_id,
        "image": str(sample.screenshot_path),
        "resolution": list(sample.resolution),
        "instruction": INSTRUCTION_TEMPLATE.format(description=description),
        # 输出格式与推理时的解析路径一致，模型不必学两套
        "answer": json.dumps({"x": nx, "y": ny}, ensure_ascii=False),
        "point_norm": [nx, ny],
        "point_px": [point.x, point.y],
        "source": sample.source_dataset,
    }
    if sample.bbox is not None:
        record["bbox_px"] = [
            sample.bbox.left,
            sample.bbox.top,
            sample.bbox.right,
            sample.bbox.bottom,
        ]
    return record


def convert(
    split_file: Path = SPLIT_FILE,
    out_dir: Path = OUT_DIR,
    dataset: str = "screenagent",
    check_only: bool = False,
) -> dict:
    """按冻结划分生成 train/val 两个 JSONL。返回统计。"""
    from data.loaders import LOADERS
    from data.split import FrozenSplit

    if not split_file.exists():
        raise SystemExit(f"找不到冻结划分 {split_file}。\n先跑 python scripts/prepare_datasets.py")

    # 指纹不符会在这里抛，**不要吞掉它** —— 拿一份被改过的划分去训练，
    # 后面所有结论都不可信，而且事后查不出来
    split = FrozenSplit.read(split_file)
    leak = split.leakage()
    if leak:
        raise SystemExit(f"划分有泄漏：{len(leak)} 个样本同时在 train 与 val。拒绝生成。")

    loader_cls = LOADERS.get(dataset)
    if loader_cls is None:
        raise SystemExit(f"没有名为 {dataset!r} 的装载器。可用：{'、'.join(LOADERS)}")
    loader = loader_cls()
    ready, why = loader.available()
    if not ready:
        raise SystemExit(f"{dataset} 数据未就绪：{why}")

    wanted = {"train": set(split.train_ids), "val": set(split.val_ids)}
    buckets: dict[str, list] = {"train": [], "val": []}
    dropped: Counter = Counter()
    seen = 0

    for sample in loader.load():
        target = None
        for name, ids in wanted.items():
            if sample.sample_id in ids:
                target = name
                break
        if target is None:
            continue
        seen += 1
        record = build_record(sample)
        if record is None:
            dropped[target] += 1
            continue
        buckets[target].append(record)

    stats = {
        "split_file": str(split_file),
        "fingerprint": split.fingerprint(),
        "seed": split.seed,
        "group_by": split.group_by,
        "expected": {"train": split.train_size, "val": split.val_size},
        "built": {k: len(v) for k, v in buckets.items()},
        "dropped": dict(dropped),
        "matched_in_dataset": seen,
        "coord_space": list(COORD_SPACE),
    }

    if check_only:
        return stats

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, records in buckets.items():
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        stats.setdefault("files", {})[name] = str(path)

    (out_dir / "meta.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return stats


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="生成 grounding 微调数据（M3 任务 1）")
    parser.add_argument("--split", default=str(SPLIT_FILE))
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--dataset", default="screenagent")
    parser.add_argument("--check", action="store_true", help="只核对划分与条数，不写文件")
    args = parser.parse_args()

    stats = convert(
        split_file=Path(args.split),
        out_dir=Path(args.out),
        dataset=args.dataset,
        check_only=args.check,
    )

    print("=" * 70)
    print("grounding 微调数据构建")
    print("=" * 70)
    print(f"  划分文件  {stats['split_file']}")
    print(f"  指纹      {stats['fingerprint']}    种子 {stats['seed']}")
    print(f"  划分单位  {stats['group_by']}")
    print(f"  坐标空间  {stats['coord_space'][0]}×{stats['coord_space'][1]}")
    print()
    for name in ("train", "val"):
        expected = stats["expected"][name]
        built = stats["built"][name]
        drop = stats["dropped"].get(name, 0)
        mark = "" if built + drop == expected else "  **与划分条数不符**"
        print(f"  {name:<6}期望 {expected:>4}    生成 {built:>4}    丢弃 {drop:>3}{mark}")
    if stats["dropped"]:
        print("\n  丢弃原因：描述为空或缺坐标。**不填占位符**——「找到：」后面")
        print("  什么都没有的样本会教模型在缺信息时硬猜坐标。")

    total = stats["built"]["train"]
    if total < 1000:
        print(f"\n  **训练集只有 {total} 条**，大纲定的是 3000-4000。")
        print("  这是已知的既成事实（M2 前置件①实测），补充来源方案见")
        print("  docs/m3-prereq/README.md，须在 M3 开工前定案。")

    if not args.check:
        print(f"\n  已写入 {args.out}/train.jsonl、val.jsonl、meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
