"""基础数据类型测试：BBox 几何运算与双通道融合去重。"""

from __future__ import annotations

import pytest

from perception.types import BBox, ElementSource, Point, UIElement, dedupe_by_iou

# --------------------------------------------------------------------- #
# Point
# --------------------------------------------------------------------- #


def test_point_rejects_float() -> None:
    """像素是离散的。允许浮点会让误差在多次转换中累积，也让验收标准难判定。"""
    with pytest.raises(TypeError):
        Point(1.5, 2)  # type: ignore[arg-type]


def test_point_is_hashable() -> None:
    assert len({Point(1, 2), Point(1, 2)}) == 1


# --------------------------------------------------------------------- #
# BBox
# --------------------------------------------------------------------- #


def test_bbox_width_height_are_open_interval() -> None:
    """right/bottom 为开区间，width 直接相减即可，不需要 +1。"""
    box = BBox(10, 20, 110, 70)
    assert (box.width, box.height) == (100, 50)
    assert box.area == 5000


def test_bbox_center() -> None:
    assert BBox(0, 0, 100, 50).center == Point(50, 25)


def test_bbox_rejects_inverted() -> None:
    with pytest.raises(ValueError):
        BBox(100, 0, 10, 50)


def test_bbox_from_xywh_matches_uia_convention() -> None:
    """UIA 的 BoundingRectangle 给的是左上角 + 宽高。"""
    assert BBox.from_xywh(10, 20, 100, 50) == BBox(10, 20, 110, 70)


def test_bbox_contains_respects_open_interval() -> None:
    box = BBox(0, 0, 10, 10)
    assert box.contains(Point(0, 0))
    assert box.contains(Point(9, 9))
    assert not box.contains(Point(10, 10))


def test_iou_identical_is_one() -> None:
    box = BBox(0, 0, 100, 100)
    assert box.iou(box) == pytest.approx(1.0)


def test_iou_disjoint_is_zero() -> None:
    assert BBox(0, 0, 10, 10).iou(BBox(50, 50, 60, 60)) == 0.0


def test_iou_touching_edges_is_zero() -> None:
    """仅边界相接不算重叠——开区间语义下它们没有共同像素。"""
    assert BBox(0, 0, 10, 10).iou(BBox(10, 0, 20, 10)) == 0.0


def test_iou_half_overlap() -> None:
    a = BBox(0, 0, 100, 100)
    b = BBox(50, 0, 150, 100)
    # 交 5000，并 15000
    assert a.iou(b) == pytest.approx(1 / 3)


# --------------------------------------------------------------------- #
# 双通道融合去重
# --------------------------------------------------------------------- #


def _elem(box: BBox, source: ElementSource, text: str = "") -> UIElement:
    return UIElement(bbox=box, source=source, text=text)


def test_uia_wins_over_overlapping_ocr() -> None:
    """UIA 框住整个可点击控件，OCR 只框住上面的文字。

    重叠时必须保留 UIA——点 OCR 小框的中心可能落在文字上而非按钮的有效
    响应区，尤其是带图标的按钮。
    """
    button = _elem(BBox(100, 100, 300, 140), ElementSource.UIA, "保存")
    text_only = _elem(BBox(150, 110, 250, 130), ElementSource.OCR, "保存")

    kept = dedupe_by_iou([text_only, button], iou_threshold=0.1)
    assert len(kept) == 1
    assert kept[0].source is ElementSource.UIA


def test_ocr_kept_when_uia_misses_it() -> None:
    """UIA 抓不到自绘控件（Electron / 游戏 / Canvas），此时 OCR 是唯一来源。"""
    uia = _elem(BBox(0, 0, 100, 40), ElementSource.UIA, "文件")
    ocr = _elem(BBox(500, 300, 600, 330), ElementSource.OCR, "自绘按钮")

    kept = dedupe_by_iou([uia, ocr])
    assert len(kept) == 2


def test_dedupe_assigns_reading_order_index() -> None:
    """编号按从上到下、从左到右，调试图上才好对照。"""
    bottom = _elem(BBox(0, 500, 50, 530), ElementSource.OCR, "下")
    top_right = _elem(BBox(400, 10, 450, 40), ElementSource.OCR, "右上")
    top_left = _elem(BBox(10, 10, 60, 40), ElementSource.OCR, "左上")

    kept = dedupe_by_iou([bottom, top_right, top_left])
    assert [e.text for e in kept] == ["左上", "右上", "下"]
    assert [e.index for e in kept] == [1, 2, 3]


def test_dedupe_threshold_is_respected() -> None:
    a = _elem(BBox(0, 0, 100, 100), ElementSource.OCR, "a")
    b = _elem(BBox(50, 0, 150, 100), ElementSource.OCR, "b")  # IoU = 1/3

    assert len(dedupe_by_iou([a, b], iou_threshold=0.5)) == 2
    assert len(dedupe_by_iou([a, b], iou_threshold=0.2)) == 1


def test_dedupe_prefers_larger_box_within_same_source() -> None:
    """同来源重叠时保留大框——同样是为了点在有效响应区内。"""
    big = _elem(BBox(0, 0, 200, 100), ElementSource.OCR, "big")
    small = _elem(BBox(10, 10, 60, 40), ElementSource.OCR, "small")

    kept = dedupe_by_iou([small, big], iou_threshold=0.05)
    assert len(kept) == 1
    assert kept[0].text == "big"


def test_dedupe_empty_input() -> None:
    assert dedupe_by_iou([]) == []


# --------------------------------------------------------------------- #
# UIElement
# --------------------------------------------------------------------- #


def test_click_point_is_box_center() -> None:
    element = _elem(BBox(100, 100, 300, 140), ElementSource.UIA, "保存")
    assert element.click_point == Point(200, 120)


def test_label_truncates_long_text() -> None:
    element = UIElement(
        bbox=BBox(0, 0, 10, 10), source=ElementSource.OCR, text="很长" * 30, index=7
    )
    label = element.label()
    assert label.startswith("7.")
    assert len(label) <= 23


def test_label_falls_back_to_control_type() -> None:
    element = UIElement(
        bbox=BBox(0, 0, 10, 10), source=ElementSource.UIA, text="", control_type="ButtonControl"
    )
    assert "ButtonControl" in element.label()
