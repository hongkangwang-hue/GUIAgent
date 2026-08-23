"""模式 B 可行性验证 —— M2 任务 5 / 验收标准 9，同时是 M3 前置件④。

## 要回答的问题

模式 B 让规划模型只输出「点什么元素」的自然语言描述，由本地 grounding
模型负责转成坐标。这套设计成立的**前提**是：规划模型能稳定产出
**足够具体、能唯一定位**的描述。

如果它做不到——比如经常漏掉 `target_description` 字段，或者输出
「那个按钮」这种谁也定位不了的话——**模式 B 从设计上就不成立**，
M3 的实现方案必须改。越早知道越好，这就是本脚本存在的理由。

## 三个层次的判据，从松到严

字段在不在、能不能解析，是**机器可判**的；描述好不好，需要判断标准。
本脚本给出三层，各自的局限写在旁边：

1. **结构化输出成功率** —— JSON 可解析 且 含 `target_description`。
   完全客观。这一层过不了，后面免谈。
2. **描述具体性（启发式）** —— 是否包含可定位线索：引号内的界面文字、
   控件类型词、方位词、颜色。**这是近似判据**：它能识别出明显含糊的
   描述，但不能保证「看起来具体」的描述真的指向了正确的元素。
3. **与人工描述语料的分布对照** —— ScreenAgent 数据集里有 1168 条
   标注者手写的元素描述（`PlanAction.element`）。拿模型输出的长度与
   线索密度与之比较，回答「模型写的描述和人写的像不像」。
   **相似不等于正确**，但明显偏离（比如平均短一半）是个危险信号。

**真正的可用性只有接上 grounding 模型跑一遍才知道**，那是 M3 的事。
本脚本的结论止于「描述的形态是否支撑得起模式 B」。

## 测试素材

用 ScreenSpot-v2 的桌面截图：真实界面、公开可复现、且**自带指令与目标
元素的真值框**。真值框在这里不用于判分（模型不输出坐标），
但它保证了「这条指令在这张图上确实有一个明确的目标」。

## 用法

    python scripts/verify_mode_b.py                  # 3 场景 × 10 次
    python scripts/verify_mode_b.py --per-scene 5    # 省钱的快速版
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPORT = Path("docs/m2-mode-b-feasibility.md")
RAW = Path("docs/m2-mode-b-raw.json")

#: 控件类型词。描述里出现其一，说明模型至少说清了「这是个什么东西」。
_TYPE_WORDS = (
    "按钮",
    "输入框",
    "文本框",
    "复选框",
    "单选",
    "菜单项",
    "菜单",
    "标签页",
    "下拉",
    "图标",
    "链接",
    "选项",
    "开关",
    "滑块",
    "列表项",
    "搜索框",
    "button",
    "input",
    "checkbox",
    "menu",
    "tab",
    "icon",
    "link",
    "field",
)

#: 方位词。用于在同类元素中区分。
_POSITION_WORDS = (
    "左侧",
    "右侧",
    "顶部",
    "底部",
    "上方",
    "下方",
    "左上",
    "右上",
    "左下",
    "右下",
    "中间",
    "中央",
    "第一",
    "第二",
    "最后",
    "旁边",
    "导航栏",
    "工具栏",
    "标题栏",
    "侧边栏",
    "left",
    "right",
    "top",
    "bottom",
    "corner",
)

_COLOR_WORDS = ("红", "蓝", "绿", "黄", "黑", "白", "灰", "橙", "紫", "高亮", "蓝色")

#: 明显无法定位的描述。命中即判为不具体，不管它还写了什么。
_VAGUE = re.compile(
    r"^(那个|这个|该|the)?\s*(按钮|图标|选项|元素|控件|button|icon|item|element)\s*$"
)


def _quoted_text(text: str) -> list[str]:
    """抠出引号内的界面文字。描述里引用了屏幕上的原文，是最强的定位线索。"""
    return re.findall(r"[「『\"'“‘]([^」』\"'”’]{1,30})[」』\"'”’]", text)


def specificity(desc: str) -> dict:
    """启发式打分。返回各线索的命中情况与一个 0-4 的总分。

    **这是近似判据**，见模块文档第 2 点。它抓得住「那个按钮」这种明显
    含糊的，抓不住「写着确定的按钮」但屏幕上有三个「确定」这种情况。
    """
    text = desc.strip()
    cues = {
        "quoted_text": bool(_quoted_text(text)),
        "type_word": any(w in text for w in _TYPE_WORDS),
        "position": any(w in text for w in _POSITION_WORDS),
        "color": any(w in text for w in _COLOR_WORDS),
    }
    score = sum(cues.values())
    if _VAGUE.match(text):
        score = 0
    return {**cues, "score": score, "length": len(text), "vague": bool(_VAGUE.match(text))}


def human_corpus_stats(limit: int = 1200) -> dict | None:
    """ScreenAgent 里人工写的元素描述的分布，作为对照基线。

    拿不到语料时返回 None——**不编造基线**，报告里如实写「未取到」。
    """
    root = Path("data/raw/screenagent_repo/data/ScreenAgent")
    if not root.is_dir():
        return None
    descs: list[str] = []
    for path in root.glob("*/*/*.json"):
        if "_neg_" in path.name:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for action in payload.get("actions") or []:
            if action.get("action_type") == "PlanAction" and action.get("element"):
                descs.append(str(action["element"]))
        if len(descs) >= limit:
            break
    if not descs:
        return None
    scores = [specificity(d) for d in descs]
    return {
        "n": len(descs),
        "mean_length": sum(s["length"] for s in scores) / len(scores),
        "mean_score": sum(s["score"] for s in scores) / len(scores),
        "vague_rate": sum(s["vague"] for s in scores) / len(scores),
        "samples": descs[:5],
    }


def load_scenes(total: int, seed: int) -> list[dict]:
    """挑截图与指令，凑够 `total` 条，尽量分散在不同界面上。

    **原本按「3 个场景 × 每场景 10 条」取样，但数据不支持。**
    ScreenSpot-v2 桌面子集共 231 张图 / 334 条指令，其中 156 张图只有
    1 条指令，最多的一张也只有 5 条——没有任何一张能单独供出 10 条。

    M2 任务 5 的原文是「至少在 3 个界面场景下各测 10 次」。这里按其
    **意图**执行：保证覆盖足够多的不同界面，总量凑到 30 条
    （同时满足 M3 前置件④的「20-30 次」）。实际用了几张图会在报告里
    如实写明，不假装满足了「每场景 10 条」。
    """
    base = Path("data/raw/screenspot_v2")
    rows = json.loads((base / "screenspot_desktop_v2.json").read_text(encoding="utf-8"))
    images = base / "screenspotv2_image"

    by_image: dict[str, list[dict]] = {}
    for row in rows:
        if (images / row["img_filename"]).exists():
            by_image.setdefault(row["img_filename"], []).append(row)
    if not by_image:
        raise SystemExit(
            "找不到可用的 ScreenSpot-v2 桌面截图。"
            "先跑 python scripts/prepare_datasets.py --download --extract"
        )

    # 指令多的图排前面：同一张图上问多个不同目标，更能检验描述的区分度
    # （屏幕上有多个候选时，模型是否说得清指的是哪一个）
    ordered = sorted(by_image.items(), key=lambda kv: -len(kv[1]))
    rng = random.Random(seed)
    head = ordered[:20]
    rng.shuffle(head)

    scenes, count = [], 0
    for name, items in head + ordered[20:]:
        if count >= total:
            break
        take = items[: min(len(items), total - count)]
        scenes.append(
            {
                "image": str(images / name),
                "instructions": [r["instruction"] for r in take],
            }
        )
        count += len(take)
    return scenes


def run_batch(backend, scenes, label: str) -> list[dict]:
    """跑一批，返回记录。"""
    import cv2
    import numpy as np

    from llm.parsing import extract_json
    from perception.capture import Screenshot
    from perception.types import BBox

    records = []
    for index, scene in enumerate(scenes, start=1):
        image = cv2.imdecode(np.fromfile(scene["image"], dtype=np.uint8), cv2.IMREAD_COLOR)
        height, width = image.shape[:2]
        shot = Screenshot(image=image, region=BBox(0, 0, width, height), engine="file")
        name = Path(scene["image"]).name
        print(f"[{label} {index}/{len(scenes)}] {name[:28]}  {width}x{height}")

        for instruction in scene["instructions"]:
            start = time.perf_counter()
            try:
                text = backend.complete(instruction, shot, history=[]).text
                error = ""
            except Exception as exc:  # noqa: BLE001 —— 单条失败不该中断整轮
                text, error = "", f"{type(exc).__name__}: {exc}"[:160]

            parsed = extract_json(text) if text else None
            desc = (
                str((parsed or {}).get("target_description") or "")
                if isinstance(parsed, dict)
                else ""
            )
            record = {
                "batch": label,
                "image": name,
                "instruction": instruction,
                "raw": text[:600],
                "error": error,
                "json_ok": isinstance(parsed, dict),
                "has_field": bool(desc),
                "done": bool((parsed or {}).get("done")) if isinstance(parsed, dict) else False,
                "description": desc,
                "action": (parsed or {}).get("action", "") if isinstance(parsed, dict) else "",
                "latency_ms": round((time.perf_counter() - start) * 1000.0, 1),
                **({"cues": specificity(desc)} if desc else {}),
            }
            records.append(record)
            if record["has_field"]:
                mark = "OK  "
            elif record["done"]:
                mark = "DONE"
            elif record["json_ok"]:
                mark = "JSON"
            else:
                mark = "FAIL"
            print(f"   [{mark}] {instruction[:32]:<34} -> {desc[:46] or error[:46]}")
    return records


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="模式 B 可行性验证")
    parser.add_argument("--total", type=int, default=30, help="主实验的指令总数")
    parser.add_argument("--repeats", type=int, default=5, help="稳定性测试：同一指令重复几次")
    parser.add_argument("--provider", default="dashscope")
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()

    from llm.openai_compat import OpenAICompatBackend
    from llm.providers import load_dotenv_if_present, resolve

    load_dotenv_if_present()
    config = resolve(args.provider)
    backend = OpenAICompatBackend(config)

    from agent.prompts import load_template, render_action_reference

    template = load_template("executor_modeb_v1")
    backend.system_prompt = template.render_system(action_reference=render_action_reference())
    backend.few_shot = template.few_shot_pairs()
    backend.user_template = template.user_template

    scenes = load_scenes(args.total, args.seed)
    print("模式 B 可行性验证")
    print(f"  主实验    {args.total} 条指令，覆盖 {len(scenes)} 张不同界面")
    print(f"  稳定性    1 条指令重复 {args.repeats} 次")
    print(f"  模型      {config.model}\n")

    records = run_batch(backend, scenes, "main")

    # 稳定性：同一张图、同一条指令重复问，看描述是否一致。
    # 「能否**稳定**产出」是 M2 任务 5 的原话，而单次成功不能回答这个。
    print()
    first = scenes[0]
    repeat_scene = [
        {
            "image": first["image"],
            "instructions": [first["instructions"][0]] * args.repeats,
        }
    ]
    records += run_batch(backend, repeat_scene, "repeat")

    render(records, backend, config)
    return 0


def render(records: list[dict], backend, config) -> None:
    """汇总。**分母的口径是这份报告最容易出错的地方。**

    第一版把「没有 target_description」一律算作字段缺失，于是模型判断
    「子任务已完成」而输出 `{"done": true}` 的那些也被计入失败——
    那是**正确行为**被算成了缺陷。同理 type / key / wait 这类动作本就
    不涉及具体元素。

    因此分三类统计，字段缺失率只在「模型确实选了一个需要目标的动作」
    这个分母上计算。
    """
    total = len(records)
    json_ok = sum(r["json_ok"] for r in records)
    done = [r for r in records if r.get("done")]

    # 不涉及具体元素的动作，本就不需要 target_description
    no_target_actions = {"type", "key", "wait", "scroll", "hotkey"}
    no_target = [
        r
        for r in records
        if not r["has_field"] and not r.get("done") and r.get("action") in no_target_actions
    ]

    # 真正的分母：选了一个需要指向某元素的动作
    needs_target = [
        r
        for r in records
        if r.get("action") and r.get("action") not in no_target_actions and not r.get("done")
    ]
    got_target = [r for r in needs_target if r["has_field"]]

    print("=" * 74)
    print("模式 B 可行性验证结果")
    print("=" * 74)
    print(f"  调用总数                {total}")
    print(f"  JSON 可解析             {json_ok}/{total}  ({json_ok / total:.1%})")
    print()
    print("  按模型的选择分类：")
    print(f"    选了需指向元素的动作   {len(needs_target)}")
    print(f"    判定子任务已完成(done) {len(done)}")
    print(f"    选了无需目标的动作     {len(no_target)}")
    print()
    if needs_target:
        rate = len(got_target) / len(needs_target)
        print(f"  **字段完备率**          {len(got_target)}/{len(needs_target)}  ({rate:.1%})")
        print("    分母只算「选了需要目标的动作」那些——done 与 type/key/wait")
        print("    不需要该字段，计入分母会把正确行为算成缺陷")

    with_cues = [r for r in records if r.get("cues")]
    if with_cues:
        mean_score = sum(r["cues"]["score"] for r in with_cues) / len(with_cues)
        mean_len = sum(r["cues"]["length"] for r in with_cues) / len(with_cues)
        vague = sum(r["cues"]["vague"] for r in with_cues)
        print()
        print(f"  描述平均线索数          {mean_score:.2f} / 4")
        print(f"  描述平均长度            {mean_len:.1f} 字")
        print(f"  明显含糊的描述          {vague}/{len(with_cues)}")
        cue_counter = Counter()
        for r in with_cues:
            for key, hit in r["cues"].items():
                if key in ("quoted_text", "type_word", "position", "color") and hit:
                    cue_counter[key] += 1
        print(
            "  线索命中分布            "
            + "  ".join(f"{k}={v}" for k, v in cue_counter.most_common())
        )

    repeats = [r for r in records if r.get("batch") == "repeat" and r["has_field"]]
    if repeats:
        variants = Counter(r["description"] for r in repeats)
        print()
        print(f"  稳定性（同一指令 ×{len(repeats)}）  {len(variants)} 种不同输出")
        for desc, n in variants.most_common():
            print(f"    ×{n}  {desc[:56]}")

    lat = sorted(r["latency_ms"] for r in records if r["latency_ms"])
    if lat:
        print()
        print(
            f"  单次调用延迟            p50 {lat[len(lat) // 2]:.0f}ms  "
            f"p95 {lat[int(len(lat) * 0.95)]:.0f}ms"
        )

    human = human_corpus_stats()
    if human:
        print(f"\n  人工描述基线（ScreenAgent，n={human['n']}）")
        print(f"    平均长度 {human['mean_length']:.1f} 字   平均线索数 {human['mean_score']:.2f}")
    else:
        print("\n  [注意] 未取到 ScreenAgent 人工描述语料，本次无对照基线")

    cost = backend.get_cost()
    print(
        f"\n  本次成本 {cost.cost_cny:.4f} 元" + ("" if cost.priced else "（单价未配置，不可信）")
    )

    RAW.parent.mkdir(parents=True, exist_ok=True)
    RAW.write_text(
        json.dumps(
            {"model": config.model, "records": records, "human_baseline": human},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  原始数据已存 {RAW}")


if __name__ == "__main__":
    raise SystemExit(main())
