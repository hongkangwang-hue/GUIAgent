"""统一样本 schema —— 所有公开数据集在此收敛成同一种形状。

## 为什么必须统一

M3 要做的事有两件：拿 grounding 训练集微调，拿 ScreenSpot 做零样本评测。
两者的原始格式毫不相干——ScreenSpot 是 parquet + 内嵌 PNG + 归一化 bbox，
ScreenAgent 是一堆按会话分目录的 JSON + jpg + 点击点。如果不先收敛，
M3 那周会同时面对"训练脚本读不了评测集"和"评测脚本读不了训练集"两个问题，
而 M3 是全项目最挤的一周。

## 坐标一律存**绝对像素**

各家原始格式不同：ScreenSpot 存归一化 [0,1]，ScreenAgent 存 1024×768 下的
绝对值，模型输出又是 [0,1000)。三套并存的后果 M1 已经吃过一次亏了——
第一次真机跑 Agent 时模型给出 (275, 969)，被当成越界拒绝，实际那是归一化坐标。

所以本模块的规则是：**入库即转绝对像素，并强制同时记录 `resolution`**。
任何归一化需求都由使用方按 `resolution` 现算，不留第二份真值。

## bbox 与 point 都可以缺，但不能都缺

- ScreenSpot 给的是元素框（有 bbox，point 取中心）
- ScreenAgent 给的是鼠标落点（有 point，无 bbox）

强行给 ScreenAgent 编一个 bbox 会污染"元素尺寸分布"这张统计图；强行要求
ScreenSpot 只留点又会丢掉评测要用的框。两个字段都设为可选，清洗时校验
"至少有一个"。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

from perception.types import BBox, Point

#: 统一 schema 的字段顺序。大纲第 3 周 D1-D2 点名的九个字段在前，
#: 之后是本项目为了可追溯补的三个。写 CSV / 建 DataFrame 都以此为准。
SCHEMA_FIELDS: tuple[str, ...] = (
    # —— 大纲指定 ——
    "sample_id",
    "screenshot_path",
    "resolution",
    "instruction",
    "action_type",
    "bbox",
    "point",
    "platform",
    "app",
    # —— 本项目补充，用于追溯与划分 ——
    "source_dataset",
    "split",
    "meta",
)


class Platform(str, Enum):
    """样本所属的界面形态。

    大纲把 Mind2Web / WebArena 列为"仅抽样查看"，理由正是它们是 WEB 而本
    项目目标是 DESKTOP。这个字段让"对不对齐"从一句论断变成一个可以 groupby
    的事实。
    """

    DESKTOP = "desktop"
    MOBILE = "mobile"
    WEB = "web"
    UNKNOWN = "unknown"


class ActionType(str, Enum):
    """归一化后的动作类型。

    有意保持粗粒度。各数据集的原始标注词表互不相同（ScreenSpot 只区分
    text/icon 这种**元素类别**而非动作，ScreenAgent 有 click/drag/scroll/
    type/key 等），细分反而会造出一堆只在单个数据集里出现的类别，让
    "动作类型分布"这张图变成三个数据集各画各的。

    原始标签一律保留在 `meta["raw_action_type"]` 里，需要细分时回去取。
    """

    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    DRAG = "drag"
    SCROLL = "scroll"
    TYPE = "type"
    KEY = "key"
    WAIT = "wait"
    #: 数据集标的是"这是个什么元素"而不是"做什么动作"（ScreenSpot 即如此）。
    #: 不硬塞成 CLICK——那等于凭空断言这些元素都该被点击。
    LOCATE = "locate"
    OTHER = "other"


@dataclass
class UnifiedSample:
    """统一后的单条样本。

    一条样本 = 一张截图 + 一句指令 + 一个目标位置。多步会话（ScreenAgent）
    在装载时被拆成逐步的独立样本，会话归属记在 `meta` 里。
    """

    sample_id: str
    screenshot_path: str
    #: (width, height)，**截图的真实像素尺寸**，不是模型输入尺寸。
    resolution: tuple[int, int]
    instruction: str
    action_type: ActionType
    platform: Platform
    source_dataset: str
    #: 绝对像素的元素框。ScreenAgent 这类只有落点的样本为 None。
    bbox: BBox | None = None
    #: 绝对像素的目标点。为 None 时由 `resolve_point()` 从 bbox 中心补。
    point: Point | None = None
    app: str = ""
    #: 数据集自带的划分（test / train / …）。本项目自己的三分见 `data.split`，
    #: 那个写在独立的 split 文件里，不覆盖此字段——原始划分要留着对照。
    split: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def width(self) -> int:
        return self.resolution[0]

    @property
    def height(self) -> int:
        return self.resolution[1]

    def resolve_point(self) -> Point | None:
        """拿到可用于训练 / 评测的目标点。

        显式给出的 point 优先于 bbox 中心：ScreenAgent 的落点是标注者真实
        点过的位置，比几何中心更能代表"可点击的地方"（想想一个很长的菜单项）。
        """
        if self.point is not None:
            return self.point
        if self.bbox is not None:
            return self.bbox.center
        return None

    def normalized_point(self, scale: int = 1000) -> tuple[float, float] | None:
        """按 `resolution` 折算成 [0, scale) 的归一化坐标。

        M3 训练样本要喂给 Qwen2.5-VL，它的坐标约定就是 0-1000 归一化——
        这一点在 M1 真机验证里被实测确认过（告诉模型图高 768 或 640，
        它给出的 y 都在 96x 附近，不随声明的尺寸变化）。
        """
        p = self.resolve_point()
        if p is None or self.width <= 0 or self.height <= 0:
            return None
        return (p.x / self.width * scale, p.y / self.height * scale)

    def to_row(self) -> dict:
        """摊平成一行，供 pandas 直接建 DataFrame。

        嵌套结构（bbox / point / resolution / meta）在这里拆开或序列化，
        因为 DataFrame 里放 dataclass 会让 groupby 与绘图都没法用。
        """
        return {
            "sample_id": self.sample_id,
            "screenshot_path": self.screenshot_path,
            "width": self.width,
            "height": self.height,
            "instruction": self.instruction,
            "action_type": self.action_type.value,
            "platform": self.platform.value,
            "app": self.app,
            "source_dataset": self.source_dataset,
            "split": self.split,
            "bbox_left": self.bbox.left if self.bbox else None,
            "bbox_top": self.bbox.top if self.bbox else None,
            "bbox_right": self.bbox.right if self.bbox else None,
            "bbox_bottom": self.bbox.bottom if self.bbox else None,
            "bbox_width": self.bbox.width if self.bbox else None,
            "bbox_height": self.bbox.height if self.bbox else None,
            "bbox_area": self.bbox.area if self.bbox else None,
            "point_x": (p := self.resolve_point()) and p.x,
            "point_y": p.y if p else None,
            "has_bbox": self.bbox is not None,
        }

    def to_json_dict(self) -> dict:
        """无损序列化，用于 JSONL 落盘。`from_json_dict` 是它的逆。"""
        return {
            "sample_id": self.sample_id,
            "screenshot_path": self.screenshot_path,
            "resolution": list(self.resolution),
            "instruction": self.instruction,
            "action_type": self.action_type.value,
            "platform": self.platform.value,
            "app": self.app,
            "source_dataset": self.source_dataset,
            "split": self.split,
            "bbox": list(self.bbox.as_tuple()) if self.bbox else None,
            "point": list(self.point.as_tuple()) if self.point else None,
            "meta": self.meta,
        }

    @classmethod
    def from_json_dict(cls, payload: dict) -> UnifiedSample:
        bbox = payload.get("bbox")
        point = payload.get("point")
        return cls(
            sample_id=payload["sample_id"],
            screenshot_path=payload["screenshot_path"],
            resolution=tuple(payload["resolution"]),  # type: ignore[arg-type]
            instruction=payload.get("instruction", ""),
            action_type=ActionType(payload.get("action_type", "other")),
            platform=Platform(payload.get("platform", "unknown")),
            app=payload.get("app", ""),
            source_dataset=payload.get("source_dataset", ""),
            split=payload.get("split", ""),
            bbox=BBox(*bbox) if bbox else None,
            point=Point(*point) if point else None,
            meta=payload.get("meta", {}),
        )


# ===================================================================== #
# 落盘 / 装载
# ===================================================================== #


def write_jsonl(samples: Iterable[UnifiedSample], path: str | Path) -> int:
    """写 JSONL。逐条写而不是先攒成 list —— 装载器都是生成器，攒起来会把
    几万条样本连同 meta 一起顶在内存里。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_json_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> Iterator[UnifiedSample]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield UnifiedSample.from_json_dict(json.loads(line))


def to_dataframe(samples: Iterable[UnifiedSample]):
    """建 DataFrame。pandas 在函数内导入，让 schema 本身保持可无依赖测试。"""
    import pandas as pd

    return pd.DataFrame([s.to_row() for s in samples])


__all__ = [
    "SCHEMA_FIELDS",
    "ActionType",
    "Platform",
    "UnifiedSample",
    "asdict",
    "read_jsonl",
    "to_dataframe",
    "write_jsonl",
]
