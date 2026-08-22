"""ScreenAgent —— M3 的 grounding 训练样本来源。

## 数据不在 HuggingFace 上

HF 的 `niurl/ScreenAgent` 只有 CogAgent 格式的权重 zip。**标注数据在作者的
GitHub 仓库里**（`niuzaisheng/ScreenAgent`，`data/ScreenAgent/`），train 是
按会话分的目录树，test 是一个 50MB 的 `test.zip`。这一点值得写进报告：
按 HF 名称去找会一无所获。

许可证：数据集 Apache-2.0（代码 MIT，模型权重走 CogVLM 协议——三者不同，
《模型与数据集许可证对照表》要分开记）。

## 三个必须知道的格式细节

**1. train 与 test 的文件命名不一样。**
train 是 `<时间戳>_translate.json`，test 是 `<时间戳>.json`。两者都还有
`_neg_plan` / `_neg_eval` 后缀的负样本（RLHF 用的 reject 回答），共 1264 个，
**这些不是标注，是模型的错误回答**，必须排除。只按 `*.json` 通配会把负样本
一起当成训练数据装进来。

**2. `mouse_position` 用 `width` / `height` 当 x / y 的键名。**
`{"mouse_position": {"width": 543, "height": 133}}` 里 width 是 x、height 是 y，
和字面意思毫无关系。这里翻译成 `Point(x, y)` 之后不再让这个命名外泄。

**3. `clickable_area`（元素框）只有 test 划分里有。**
实测：test 的 218 个带落点鼠标动作全都有框；train 的 757 个鼠标动作**一个也没有**。
也就是说 ScreenAgent 能给 M3 的训练监督信号只有点，没有框。

## 一个动作一条样本

按动作展开而不是按步，是因为 grounding 训练要的正是"一句描述 → 一个坐标"。

## 为什么 `PlanAction.element` **不能**用作 grounding 指令

这一条是踩过坑之后写的。`element` 字段看上去正是我们要的东西——标注者
手写的自然语言描述，形如 "Choose the blue color from the color palette"、
"点击save"。很自然会想把它当成鼠标动作的 `instruction`。

**但数据结构不支持这么做。** 实测确认了三件事：

1. **Plan 与 Mouse 从不出现在同一步里。** ScreenAgent 的一步是单一类型的：
   要么整步都是 PlanAction，要么整步都是 MouseAction。实测 716 个带坐标的
   鼠标动作，**同一步内前方有 PlanAction 的是 0 个**。典型会话长这样：

       step1 (255913.jpg)  PlanAction ×2   规划整个任务的几个子目标
       step2 (140419.jpg)  MouseAction click(487,193) + 输入 + 回车
       step3 (168194.jpg)  EvaluateSubTaskAction  sub_task_success

2. **就算跨步去找最近的 Plan，它描述的也是另一张截图。** 上例里 Plan 写在
   `255913.jpg` 上，点击发生在 `140419.jpg` 上。把前者的描述配后者的坐标，
   等于给模型一对错配的监督信号。

3. **`element` 中英混杂且没有语言变体。** 实测 1168 条里 73% 是中文、
   27% 是英文，而这个字段不像 `task_prompt` 那样有 `_en` / `_zh` 两套。
   用它当指令会让 `language` 选项形同虚设——选 en 的训练集里照样混进
   七成中文。

所以本装载器**一律用 `task_prompt` 作为 `instruction`**，语言由 `language`
选项决定，`meta["instruction_source"]` 恒为 `"task_prompt"`。

**这个结论对 M3 是坏消息，必须记清楚**：ScreenAgent 的 716 个点击，能配上
的只有整体任务目标（"在网上查冯诺依曼的资料"），而且 716 条里只有 **169 条
不重复**的任务目标。作为 grounding 监督信号，这批数据比条数看起来弱得多。

`element` 文本仍然保留在 PlanAction 样本的 `meta["plan_element"]` 里——
它对 grounding 没用，但**是模式 B `target_description` 的现成参考语料**
（M3 前置件④）。
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from pathlib import Path

from data.loaders.base import DatasetLoader
from data.schema import ActionType, Platform, UnifiedSample
from perception.types import BBox, Point

logger = logging.getLogger(__name__)

#: 负样本（RLHF 的 reject 回答）。**不是标注数据。**
_NEGATIVE = re.compile(r"_neg_(plan|eval)\.json$")

#: ScreenAgent 的 mouse_action_type → 统一动作类型。
#: `down` / `up` 是拖拽的两端，单独出现时归入 DRAG；`move` 是纯移动（悬停），
#: 它带坐标所以对 grounding 有用，但不该和 click 混为一谈。
_MOUSE_ACTIONS = {
    "click": ActionType.CLICK,
    "double_click": ActionType.DOUBLE_CLICK,
    "right_click": ActionType.RIGHT_CLICK,
    "move": ActionType.OTHER,
    "drag": ActionType.DRAG,
    "down": ActionType.DRAG,
    "up": ActionType.DRAG,
    "scroll_up": ActionType.SCROLL,
    "scroll_down": ActionType.SCROLL,
}

_KEYBOARD_ACTIONS = {
    "text": ActionType.TYPE,
    "press": ActionType.KEY,
}


def _point_of(payload: dict | None) -> Point | None:
    """`{"width": x, "height": y}` → `Point(x, y)`。见模块文档第 2 点。"""
    if not isinstance(payload, dict):
        return None
    x, y = payload.get("width"), payload.get("height")
    if x is None or y is None:
        return None
    return Point(int(round(float(x))), int(round(float(y))))


def _bbox_of(payload: dict | None) -> BBox | None:
    """`clickable_area` → BBox。只有 test 划分有这个字段。"""
    if not isinstance(payload, dict):
        return None
    upper = _point_of(payload.get("upper_left_position"))
    lower = _point_of(payload.get("lower_right_position"))
    if upper is None or lower is None:
        return None
    if lower.x < upper.x or lower.y < upper.y:
        return None
    return BBox(upper.x, upper.y, lower.x, lower.y)


class ScreenAgentLoader(DatasetLoader):
    """装载 ScreenAgent 的 train / test 两个划分。"""

    name = "screenagent"

    #: 仓库整体 clone 下来，数据在这个子路径下
    _DATA_SUBPATH = "data/ScreenAgent"

    def __init__(self, root: str | Path = "data/raw", **kwargs) -> None:
        super().__init__(root, **kwargs)
        #: 语言。数据集自带 en / zh 两套文本，训练用哪套要显式选，
        #: 混着来会让模型同时见到两种语言的同一批样本。
        self.language: str = kwargs.get("language", "en")

    @property
    def dataset_dir(self) -> Path:
        return self.root / "screenagent_repo" / self._DATA_SUBPATH

    def available(self) -> tuple[bool, str]:
        if not self.dataset_dir.is_dir():
            return False, (
                f"未找到 {self.dataset_dir}。ScreenAgent 数据集不在 HuggingFace 上，"
                "需 clone github.com/niuzaisheng/ScreenAgent，"
                "见 `python scripts/prepare_datasets.py --download`"
            )
        if not (self.dataset_dir / "train").is_dir():
            return False, f"{self.dataset_dir} 下没有 train 目录"
        if not (self.dataset_dir / "test").is_dir():
            return False, (
                f"test 未解压：{self.dataset_dir / 'test.zip'} → "
                f"{self.dataset_dir / 'test'}（用 --extract 解压）"
            )
        return True, ""

    def load(self) -> Iterator[UnifiedSample]:
        for split in ("train", "test"):
            split_dir = self.dataset_dir / split
            for path in sorted(split_dir.glob("*/*.json")):
                if _NEGATIVE.search(path.name):
                    self._skip("RLHF 负样本")
                    continue
                yield from self._load_step(path, split)

    # ----------------------------------------------------------------- #

    def _load_step(self, path: Path, split: str) -> Iterator[UnifiedSample]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._skip("标注文件无法解析")
            return

        width = payload.get("video_width")
        height = payload.get("video_height")
        if not width or not height:
            self._skip("缺少分辨率")
            return

        image_name = payload.get("saved_image_name")
        if not image_name:
            self._skip("缺少截图文件名")
            return
        image_path = path.parent / "images" / image_name
        if not image_path.exists():
            self._skip("截图文件不存在")
            return

        suffix = "_zh" if self.language == "zh" else "_en"
        task_prompt = payload.get(f"task_prompt{suffix}") or payload.get("task_prompt") or ""
        session_id = payload.get("session_id") or path.parent.name
        stem = path.stem.replace("_translate", "")

        for index, action in enumerate(payload.get("actions") or []):
            kind = action.get("action_type")

            parsed = self._parse_action(action)
            if parsed is None:
                self._skip(f"未识别的动作类型：{kind}")
                continue
            action_type, point, bbox = parsed

            # 指令一律取任务目标。**不用 `PlanAction.element`**——理由见模块
            # 文档「为什么 element 不能用作 grounding 指令」，一句话说是：
            # Plan 和 Mouse 不在同一步、描述的不是同一张截图、且中英混杂。
            instruction, source = task_prompt, "task_prompt"

            # element 文本对 grounding 没用，但别丢——它是模式 B
            # `target_description` 的参考语料（M3 前置件④）。
            plan_element = str(action.get("element") or "") if kind == "PlanAction" else ""

            yield UnifiedSample(
                sample_id=f"screenagent-{split}-{session_id}-{stem}-{index}",
                screenshot_path=str(image_path),
                resolution=(int(width), int(height)),
                instruction=instruction,
                action_type=action_type,
                platform=Platform.DESKTOP,
                source_dataset=self.name,
                bbox=bbox,
                point=point,
                app="",  # 数据集未标注具体应用，留空而不是猜
                split=split,
                meta={
                    "session_id": session_id,
                    "task_prompt": task_prompt,
                    "instruction_source": source,
                    "raw_action_type": kind,
                    "raw_action_subtype": (
                        action.get("mouse_action_type") or action.get("keyboard_action_type") or ""
                    ),
                    "action_index": index,
                    "language": self.language,
                    #: 仅 PlanAction 有。见模块文档——不作指令，留作模式 B 语料
                    **({"plan_element": plan_element} if plan_element else {}),
                },
            )

    @staticmethod
    def _parse_action(action: dict) -> tuple[ActionType, Point | None, BBox | None] | None:
        kind = action.get("action_type")

        if kind == "MouseAction":
            subtype = action.get("mouse_action_type", "")
            mapped = _MOUSE_ACTIONS.get(subtype)
            if mapped is None:
                return None
            return (
                mapped,
                _point_of(action.get("mouse_position")),
                _bbox_of(action.get("clickable_area")),
            )

        if kind == "KeyboardAction":
            subtype = action.get("keyboard_action_type", "")
            mapped = _KEYBOARD_ACTIONS.get(subtype)
            return (mapped, None, None) if mapped else None

        if kind == "WaitAction":
            return ActionType.WAIT, None, None

        # PlanAction / EvaluateSubTaskAction：不是键鼠动作，但确实是数据集
        # 标注的一部分（ScreenAgent 标的是"规划-执行-评估"三段循环）。
        # 保留成 OTHER，动作类型分布那张图才如实反映这个结构——这恰好也是
        # M4 Reflector 的设计参考。
        if kind in ("PlanAction", "EvaluateSubTaskAction"):
            return ActionType.OTHER, None, None

        return None
