"""召回率评测与 OCR 对照实验的计算逻辑测试。

这两处的 bug **不会报错，只会让验收数字悄悄错掉**——召回率算高了没人
发现，算低了会导向错误的优化方向。因此判据逻辑必须单独测。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from eval_recall import ImageReport, aggregate, evaluate_image  # noqa: E402
from ocr_benchmark import char_accuracy, classify, is_cjk  # noqa: E402


def _detections(elements, region=(0, 0, 1920, 1080)) -> dict:
    return {
        "region": list(region),
        "elements": [
            {
                "bbox": list(e["bbox"]),
                "click_point": e.get(
                    "click_point",
                    [
                        (e["bbox"][0] + e["bbox"][2]) // 2,
                        (e["bbox"][1] + e["bbox"][3]) // 2,
                    ],
                ),
                "source": e.get("source", "ocr"),
                "text": e.get("text", ""),
            }
            for e in elements
        ],
    }


def _gt(elements) -> dict:
    return {
        "image": "test.png",
        "elements": [
            {"label": e.get("label", ""), "type": e.get("type", "button"), "bbox": list(e["bbox"])}
            for e in elements
        ],
    }


# ===================================================================== #
# 点击点判据 —— 主判据
# ===================================================================== #


def test_small_ocr_box_inside_button_counts_as_hit() -> None:
    """OCR 只框住按钮上的文字，IoU 很低，但点击点落在按钮内 —— 算命中。

    这正是选点击点判据而非 IoU 的理由：对 Agent 来说框重合多少不重要，
    点下去能不能点中才重要。
    """
    gt = _gt([{"bbox": (100, 100, 300, 160), "label": "保存"}])
    detections = _detections([{"bbox": (150, 115, 250, 145), "source": "ocr", "text": "保存"}])

    report = evaluate_image(gt, detections, iou_threshold=0.5)
    match = report.matches[0]
    assert match.hit_by_click
    assert not match.hit_by_iou       # IoU 只有 0.25 左右
    assert match.best_iou < 0.5


def test_box_spanning_two_buttons_misses_by_click_point() -> None:
    """横跨两个按钮的大框，IoU 可能达标，但中心点落在两者之间的缝隙上。

    这种"框看着对、点下去点空"的情况，IoU 判据抓不出来。
    """
    gt = _gt([{"bbox": (0, 0, 90, 40)}])
    # 大框覆盖 0-210，中心点 (105, 20) 落在两个按钮中间的空白处
    detections = _detections([{"bbox": (0, 0, 210, 40), "click_point": [105, 20]}])

    match = evaluate_image(gt, detections, iou_threshold=0.4).matches[0]
    assert not match.hit_by_click
    assert match.hit_by_iou           # IoU ≈ 0.43，按传统判据算命中


def test_click_point_on_boundary_is_miss() -> None:
    """BBox 的 right/bottom 是开区间，边界上的点不算在内。"""
    gt = _gt([{"bbox": (0, 0, 100, 100)}])
    detections = _detections([{"bbox": (0, 0, 200, 200), "click_point": [100, 100]}])
    assert not evaluate_image(gt, detections, 0.5).matches[0].hit_by_click


def test_completely_missed_element() -> None:
    gt = _gt([{"bbox": (500, 500, 600, 540), "label": "漏检的按钮"}])
    detections = _detections([{"bbox": (0, 0, 100, 40)}])

    match = evaluate_image(gt, detections, 0.5).matches[0]
    assert not match.hit_by_click
    assert match.best_iou == 0.0
    assert match.label == "漏检的按钮"


# ===================================================================== #
# 坐标系对齐 —— 最容易埋雷的地方
# ===================================================================== #


def test_screen_coordinates_are_converted_to_image_coordinates() -> None:
    """真值是图像相对坐标，检测结果是屏幕绝对坐标，比对前必须减掉区域原点。

    截全屏时两者恰好重合，这个 bug 在开发期不会暴露；一旦改成截区域或
    副显示器，所有元素都会算成漏检。
    """
    region = (500, 300, 1300, 900)   # 非零原点
    gt = _gt([{"bbox": (10, 20, 110, 60)}])                    # 图像坐标
    detections = _detections(
        [{"bbox": (510, 320, 610, 360)}], region=region        # 同一元素的屏幕坐标
    )

    assert evaluate_image(gt, detections, 0.5).matches[0].hit_by_click


def test_forgetting_offset_would_fail() -> None:
    """反证：不减原点的话，同一个元素会被算成漏检。"""
    gt = _gt([{"bbox": (10, 20, 110, 60)}])
    detections = _detections([{"bbox": (510, 320, 610, 360)}], region=(0, 0, 1920, 1080))
    assert not evaluate_image(gt, detections, 0.5).matches[0].hit_by_click


# ===================================================================== #
# 分项统计
# ===================================================================== #


def test_hit_source_is_recorded() -> None:
    """按来源分项统计是 M1 验收标准 4 的明确要求。"""
    gt = _gt([{"bbox": (0, 0, 100, 40)}, {"bbox": (200, 0, 300, 40)}])
    detections = _detections(
        [
            {"bbox": (0, 0, 100, 40), "source": "uia"},
            {"bbox": (200, 0, 300, 40), "source": "ocr"},
        ]
    )

    report = evaluate_image(gt, detections, 0.5)
    assert {m.hit_source for m in report.matches} == {"uia", "ocr"}
    assert report.recall("click", "uia") == 0.5
    assert report.recall("click", "ocr") == 0.5
    assert report.recall("click") == 1.0


def test_recall_is_zero_when_no_ground_truth() -> None:
    assert ImageReport(name="x", total_gt=0).recall() == 0.0


def test_aggregate_across_images() -> None:
    reports = [
        evaluate_image(
            _gt([{"bbox": (0, 0, 100, 40), "type": "button"}]),
            _detections([{"bbox": (0, 0, 100, 40), "source": "uia"}]),
            0.5,
        ),
        evaluate_image(
            _gt([{"bbox": (0, 0, 100, 40), "type": "input"}]),
            _detections([{"bbox": (500, 500, 600, 540)}]),
            0.5,
        ),
    ]
    summary = aggregate(reports)
    assert summary["images"] == 2
    assert summary["total_gt"] == 2
    assert summary["recall_click"] == 0.5
    assert summary["recall_by_type"]["button"]["recall"] == 1.0
    assert summary["recall_by_type"]["input"]["recall"] == 0.0


def test_aggregate_handles_empty() -> None:
    assert aggregate([])["total_gt"] == 0


# ===================================================================== #
# OCR 准确率计算
# ===================================================================== #


@pytest.mark.parametrize("char,expected", [("中", True), ("A", False), ("1", False), ("，", False)])
def test_is_cjk(char: str, expected: bool) -> None:
    assert is_cjk(char) is expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("确定", "chinese"),
        ("Save", "english"),
        ("保存 Save", "mixed"),
        ("确定123", "mixed"),
        ("2026", "english"),
    ],
)
def test_classify(text: str, expected: str) -> None:
    assert classify(text) == expected


def test_char_accuracy_exact_match() -> None:
    assert char_accuracy("确定", "确定") == 1.0


def test_char_accuracy_partial_beats_total_failure() -> None:
    """认错一个字和整条没认出来必须区分开。

    用精确匹配的话两者都记 0 分，会掩盖真实差距——而"20 个字对 19 个"
    在实际使用中是可用的，"一个都没认出来"是不可用的。
    """
    partial = char_accuracy("确宝", "确定")   # 2 个字对 1 个
    nothing = char_accuracy("", "确定")       # 一个都没认出来
    assert nothing == 0.0
    assert partial == pytest.approx(0.5)
    assert nothing < partial < 1.0


def test_char_accuracy_empty_truth() -> None:
    assert char_accuracy("", "") == 1.0
    assert char_accuracy("多余", "") == 0.0


def test_char_accuracy_never_negative() -> None:
    """预测比真值长很多时编辑距离会超过真值长度，结果必须夹到 0。"""
    assert char_accuracy("这是一段完全不相干的很长的文本", "OK") == 0.0
