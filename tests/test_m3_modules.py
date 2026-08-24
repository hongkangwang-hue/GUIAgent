"""M3 三个新模块的纯函数测试。

覆盖 `finetune.dataset` 的坐标归一化与样本构建，以及
`grounding.local_vlm` 的坐标解析。

**训练与推理的坐标空间一致性是这里最要紧的一条。** 不一致时 loss 会
正常下降、评测分数却莫名其妙地低，因为模型学的坐标系和评测换算用的
不是一个。这种错误只能靠单测钉住，跑训练是发现不了的。
"""

from __future__ import annotations

import types

from finetune.dataset import COORD_SPACE, build_record, describe, normalize_point
from grounding.local_vlm import MODEL_SPACE, parse_point
from perception.types import BBox, Point

#: 默认落点。提到模块级是为了避开「函数默认值里做调用」这条 lint，
#: 顺带也让「所有用例共享同一个基准点」这件事显式化。
_CENTER = Point(960, 540)


def _sample(
    sample_id="s1",
    instruction="点击左上角的文件菜单",
    resolution=(1920, 1080),
    point=_CENTER,
    bbox=None,
    meta=None,
):
    """造一个够用的假样本。不 import UnifiedSample 是为了不牵扯整条数据链。"""
    return types.SimpleNamespace(
        sample_id=sample_id,
        instruction=instruction,
        resolution=resolution,
        point=point,
        bbox=bbox,
        screenshot_path="/tmp/x.png",
        source_dataset="screenagent",
        meta=meta or {},
    )


class TestCoordSpaceConsistency:
    def test_训练与推理用同一个坐标空间(self):
        """`finetune.dataset.COORD_SPACE` 必须等于 `local_vlm.MODEL_SPACE`。

        这条是整个 M3 最容易犯又最难查的错误的护栏。两边不一致时：
        训练正常、loss 正常下降、adapter 正常保存，只有评测分数低得
        莫名其妙 —— 而没人会想到去比这两个常量。
        """
        assert COORD_SPACE == MODEL_SPACE


class TestNormalizePoint:
    def test_中心点(self):
        assert normalize_point(960, 540, (1920, 1080)) == (500, 500)

    def test_原点(self):
        assert normalize_point(0, 0, (1920, 1080)) == (0, 0)

    def test_右下角不越界(self):
        """归一化空间是左闭右开的 [0,1000)。

        坐标落在图像最右一列时 `x/width*1000` 会等于 1000，越界。
        几千条样本里这种边界值必然出现，夹到 999 而不是让它溢出。
        """
        nx, ny = normalize_point(1919, 1079, (1920, 1080))
        assert nx == 999
        assert ny == 999
        assert nx < COORD_SPACE[0]
        assert ny < COORD_SPACE[1]

    def test_各轴独立(self):
        assert normalize_point(400, 300, (800, 600)) == (500, 500)

    def test_分辨率非法直接抛(self):
        """宁可抛也不要静默产生垃圾样本 —— 那会污染整个训练集。"""
        import pytest

        with pytest.raises(ValueError):
            normalize_point(1, 1, (0, 100))


class TestDescribe:
    def test_优先用指令文本(self):
        assert describe(_sample(instruction="点击保存按钮")) == "点击保存按钮"

    def test_指令为空时退回元素类型(self):
        sample = _sample(instruction="", meta={"element_kind": "icon"})
        assert describe(sample) == "icon"

    def test_都没有时返回空(self):
        assert describe(_sample(instruction="", meta={})) == ""


class TestBuildRecord:
    def test_基本构建(self):
        record = build_record(_sample())
        assert record is not None
        assert record["point_norm"] == [500, 500]
        assert record["point_px"] == [960, 540]
        assert "文件菜单" in record["instruction"]
        assert record["answer"] == '{"x": 500, "y": 500}'

    def test_没有_point_时用_bbox_中心(self):
        sample = _sample(point=None, bbox=BBox(100, 100, 300, 300))
        record = build_record(sample)
        assert record is not None
        assert record["point_px"] == [200, 200]
        assert record["bbox_px"] == [100, 100, 300, 300]

    def test_描述为空的样本丢掉(self):
        """**不填占位符。**

        「找到：」后面什么都没有的训练样本，会教模型在缺信息时也硬猜
        坐标 —— 而那正是 M2 实测里最麻烦的行为之一（凭空断言屏幕上有
        不存在的元素）。宁可少 569 条里的几条。
        """
        assert build_record(_sample(instruction="", meta={})) is None

    def test_既无_point_也无_bbox_丢掉(self):
        assert build_record(_sample(point=None, bbox=None)) is None

    def test_分辨率缺失丢掉(self):
        assert build_record(_sample(resolution=(0, 0))) is None


class TestParsePoint:
    def test_标准_json(self):
        point = parse_point('{"x": 456, "y": 310}')
        assert point == Point(456, 310)

    def test_json_被截断也能救回来(self):
        """微调后的模型格式遵从度是可预期的薄弱环节 —— 训练集只有 569 条。

        一个能解析出坐标的回答，不该因为少个花括号被判成定位失败。
        """
        assert parse_point('{"x":456,"y":310') == Point(456, 310)

    def test_裸的数字对(self):
        assert parse_point("坐标是 (456, 310)") == Point(456, 310)
        assert parse_point("456，310") == Point(456, 310)

    def test_夹在文字里的_json(self):
        assert parse_point('好的，答案是 {"x": 12, "y": 34} 。') == Point(12, 34)

    def test_没有坐标返回_None(self):
        assert parse_point("我找不到这个元素") is None
        assert parse_point("") is None

    def test_负数不被当成坐标丢掉(self):
        """解析阶段不判合法性，越界检查在 `locate()` 里做。

        两件事分开：解析回答「模型说了什么」，检查回答「说的能不能用」。
        混在一起的话，一个越界坐标会被报成「输出里没有坐标」，
        而那两种失败的修法完全不同。
        """
        assert parse_point('{"x": -5, "y": 10}') == Point(-5, 10)
