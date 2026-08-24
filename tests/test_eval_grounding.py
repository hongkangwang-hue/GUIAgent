"""`eval.grounding` 的纯函数测试。

只测不碰网络、不碰数据集的那部分：坐标换算、命中判定、尺寸分档、
断点续跑的已完成集合、汇总统计。

**坐标换算是这里最要紧的一条。** 搞错不会抛异常，只会让所有点系统性
错位，看起来像「模型定位不准」——那种错误靠跑评测发现不了，只能靠
单测钉住。
"""

from __future__ import annotations

import json

from eval.grounding import _bucket, _done_ids, inside, summarize, to_pixels
from perception.types import BBox, Point


class TestToPixels:
    def test_归一化空间换算(self):
        # qwen3-vl 输出 [0,1000)，与送多大的图无关
        assert to_pixels(500, 500, (1000, 1000), (1920, 1080)) == Point(960, 540)
        assert to_pixels(0, 0, (1000, 1000), (1920, 1080)) == Point(0, 0)

    def test_像素空间是恒等变换(self):
        """`space == size` 时不做任何缩放。

        这一条保证了「输出图像像素的模型」不需要单独的代码分支——
        少一个分支就少一处能写错的地方。
        """
        assert to_pixels(960, 540, (1920, 1080), (1920, 1080)) == Point(960, 540)

    def test_各轴独立缩放(self):
        """宽高比不同时两轴的比例不同，不能用同一个 scale。"""
        assert to_pixels(500, 500, (1000, 1000), (1920, 1080)) == Point(960, 540)
        assert to_pixels(250, 750, (1000, 1000), (800, 600)) == Point(200, 450)

    def test_搞错空间会系统性错位(self):
        """把归一化坐标当成像素用，点会挤在左上角。

        这个测试不是在测 `to_pixels` 有 bug，是把**错误用法的后果**
        钉在这里：一旦有人改默认值，这条会提醒他代价是什么。
        """
        wrong = to_pixels(500, 500, (1920, 1080), (1920, 1080))
        assert wrong == Point(500, 500)  # 而正确答案是 (960, 540)


class TestInside:
    def test_框内命中(self):
        assert inside(Point(150, 150), BBox(100, 100, 200, 200))

    def test_左上闭右下开(self):
        """与 `BBox` 的开区间语义一致，避免 off-by-one。"""
        box = BBox(100, 100, 200, 200)
        assert inside(Point(100, 100), box)
        assert not inside(Point(200, 150), box)
        assert not inside(Point(150, 200), box)

    def test_框外不命中(self):
        box = BBox(100, 100, 200, 200)
        assert not inside(Point(99, 150), box)
        assert not inside(Point(150, 99), box)


class TestBucket:
    def test_按面积分档(self):
        assert _bucket(500) == "极小 <1k"
        assert _bucket(3_000) == "小 1k-5k"
        assert _bucket(10_000) == "中 5k-20k"
        assert _bucket(50_000) == "大 >20k"

    def test_边界归入上一档(self):
        assert _bucket(1_000) == "小 1k-5k"
        assert _bucket(20_000) == "大 >20k"


class TestDoneIds:
    def test_文件不存在返回空集(self, tmp_path):
        assert _done_ids(tmp_path / "nope.jsonl") == set()

    def test_读出已完成的_id(self, tmp_path):
        path = tmp_path / "r.jsonl"
        path.write_text(
            '{"sample_id": "a"}\n{"sample_id": "b"}\n',
            encoding="utf-8",
        )
        assert _done_ids(path) == {"a", "b"}

    def test_坏行跳过不抛异常(self, tmp_path):
        """断电时最后一行天然可能残缺，**不该毁掉前面所有结果**。"""
        path = tmp_path / "r.jsonl"
        path.write_text(
            '{"sample_id": "a"}\n{"sample_id"\n{"sample_id": "b"}\n{"no_id": 1}\n',
            encoding="utf-8",
        )
        assert _done_ids(path) == {"a", "b"}


def _row(sample_id, hit, platform="desktop", kind="text", bbox=(0, 0, 100, 100), error=""):
    return {
        "sample_id": sample_id,
        "platform": platform,
        "element_kind": kind,
        "instruction": "x",
        "bbox": list(bbox),
        "raw_xy": [1, 1],
        "pixel_xy": [1, 1],
        "hit": hit,
        "error": error,
        "latency_ms": 100.0,
        "tokens": 10,
        "cost_cny": 0.0,
    }


class TestSummarize:
    def test_基本统计(self, tmp_path):
        path = tmp_path / "r.jsonl"
        rows = [_row("a", True), _row("b", False), _row("c", True)]
        path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )
        stats = summarize(path)
        assert stats["total"] == 3
        assert stats["hits"] == 2
        assert abs(stats["accuracy"] - 2 / 3) < 1e-9

    def test_调用失败不进分母(self, tmp_path):
        """API 报错的样本既不算命中也不算未命中。

        把它当未命中会低估模型能力，而失败的原因（限流、超时）跟定位
        能力无关——那是平台的问题，不该记在模型头上。
        """
        path = tmp_path / "r.jsonl"
        rows = [_row("a", True), _row("b", False, error="rate_limit: 429")]
        path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )
        stats = summarize(path)
        assert stats["total"] == 2
        assert stats["errors"] == 1
        assert stats["accuracy"] == 1.0  # 1/1，不是 1/2

    def test_分档统计(self, tmp_path):
        path = tmp_path / "r.jsonl"
        rows = [
            _row("a", True, platform="desktop", bbox=(0, 0, 10, 10)),  # 面积 100
            _row("b", False, platform="web", bbox=(0, 0, 300, 300)),  # 面积 90000
        ]
        path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )
        stats = summarize(path)
        assert stats["by_platform"]["desktop"]["acc"] == 1.0
        assert stats["by_platform"]["web"]["acc"] == 0.0
        assert stats["by_size"]["极小 <1k"]["n"] == 1
        assert stats["by_size"]["大 >20k"]["n"] == 1

    def test_空文件不除零(self, tmp_path):
        path = tmp_path / "r.jsonl"
        path.write_text("", encoding="utf-8")
        stats = summarize(path)
        assert stats["total"] == 0
        assert stats["accuracy"] == 0.0
