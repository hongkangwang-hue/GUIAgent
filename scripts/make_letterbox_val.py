"""把验证集 letterbox 成 16:9，用于验证 x 坐标塌陷是否由宽高比造成。

## 为什么要这个

`docs/m3-微调效果对比分析报告.md` §11.7.7 的实测：

    客机 1920×1080（16:9）  预测 x 中位 508，54.9% 落在中心区 [480,540]
    离线 1024×768（4:3）    预测 x 中位 425，19.2% 落在中心区

**塌陷只在客机上发生。** 疑似成因是宽高比——训练图是 4:3，客机是 16:9，
坐标归一到 0~1000 空间时 16:9 的水平方向被压缩得更厉害。

## 判据

**补黑边而不是拉伸。** 拉伸会同时改变内容的形状，那就不是单变量了；
补黑边只改画幅，内容像素一个不动。

跑完对比两组：

    原图 4:3     预测落在中心区的比例  vs  真值落在中心区的比例
    letterbox    同上

若 letterbox 后**预测的集中度远高于真值的集中度**，宽高比就是原因；
若两者一起集中（因为内容本来就被挤进了中间那条带），说明不是。

**必须拿真值做对照** —— letterbox 之后真值本来就会向中心集中
（内容占 [125, 875]），只看预测会得出假阳性。

## 用法

    python scripts/make_letterbox_val.py
    python -m eval.action --val finetune/data/val-16x9.jsonl \
        --provider selfhost --base-url http://127.0.0.1:8000/v1 --tag after-16x9
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

SRC = Path("finetune/data/val.jsonl")
DST = Path("finetune/data/val-16x9.jsonl")
IMG_DIR = Path("data/letterbox-16x9")
TARGET_RATIO = 16 / 9


def letterbox(img: Image.Image) -> tuple[Image.Image, int, int]:
    """补黑边到 16:9。返回新图与内容左上角偏移。"""
    w, h = img.size
    if w / h >= TARGET_RATIO:
        new_w, new_h = w, round(w / TARGET_RATIO)
    else:
        new_w, new_h = round(h * TARGET_RATIO), h
    canvas = Image.new("RGB", (new_w, new_h), (0, 0, 0))
    dx, dy = (new_w - w) // 2, (new_h - h) // 2
    canvas.paste(img, (dx, dy))
    return canvas, dx, dy


def _remap(x, y, ow, oh, dx, dy, nw, nh):
    """归一空间 0~1000 → 原图像素 → 加黑边偏移 → 新归一空间。

    **参数全部显式传入，不靠闭包捕获循环变量**（ruff B023）。
    """
    return (
        round((x / 1000 * ow + dx) / nw * 1000),
        round((y / 1000 * oh + dy) / nh * 1000),
    )


def main() -> int:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in SRC.open(encoding="utf-8") if line.strip()]
    print(f"读入 {len(rows)} 条")

    cache: dict[str, tuple[str, int, int, int, int]] = {}
    out = []
    for row in rows:
        src = row["image"]
        if src not in cache:
            img = Image.open(src).convert("RGB")
            new, dx, dy = letterbox(img)
            name = f"{abs(hash(src)) % (10**12):012d}.png"
            path = IMG_DIR / name
            new.save(path)
            cache[src] = (str(path).replace("\\", "/"), dx, dy, new.width, new.height)
        path, dx, dy, nw, nh = cache[src]
        ow, oh = row["resolution"]

        new_row = dict(row)
        new_row["image"] = path
        new_row["resolution"] = [nw, nh]

        # **真值坐标必须跟着变换，而且一个字段都不能漏。**
        #
        # 第一版只改了 `answer` 与 `params["point"]`，漏了 `point_norm`
        # —— 而 `eval/action.py:513` 读的正是 `point_norm`。结果是真值留在
        # 4:3 空间、预测在 16:9 空间，误差与命中率那两列全部作废。
        # 教训：**变换坐标前先查清楚下游读的是哪个字段**，不要只改看得见的那个。

        if row.get("point_norm") and len(row["point_norm"]) == 2:
            new_row["point_norm"] = list(_remap(*row["point_norm"], ow, oh, dx, dy, nw, nh))
        if row.get("point_px") and len(row["point_px"]) == 2:
            # 这个字段是像素坐标，直接加偏移
            new_row["point_px"] = [row["point_px"][0] + dx, row["point_px"][1] + dy]

        params = dict(row.get("params") or {})
        pt = params.get("point")
        if pt and len(pt) == 2:
            params["point"] = list(_remap(*pt, ow, oh, dx, dy, nw, nh))
            new_row["params"] = params
        ans = json.loads(row["answer"])
        if "x" in ans and "y" in ans:
            ans["x"], ans["y"] = _remap(ans["x"], ans["y"], ow, oh, dx, dy, nw, nh)
            new_row["answer"] = json.dumps(ans, ensure_ascii=False)
        out.append(new_row)

    with DST.open("w", encoding="utf-8", newline="\n") as fh:
        for row in out:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    first = next(iter(cache.values()))
    print(f"去重图片 {len(cache)} 张 → {IMG_DIR}")
    print(f"  例：{first[3]}x{first[4]}  左右各补 {first[1]}px 黑边")
    print(f"写出 {len(out)} 条 → {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
