"""感知流水线内部逻辑测试：预处理、OCR 解析、双通道融合、UIA 过滤、可视化。

**全部用假引擎，不加载真实 OCR 模型。** 理由有两条：

1. 加载 PaddleOCR / EasyOCR 要几秒并下载权重，会让本地跑测试变成一件
   需要下决心的事——久而久之就没人跑了。
2. 真正会出错的不是模型推理，而是**它周围的胶水代码**：坐标偏移、超分
   还原、输出格式解析、去重优先级。这些用假数据测得更准，也测得到边界。

真实模型的可用性由 `scripts/env_check.py` 负责，不属于单元测试的职责。
"""

from __future__ import annotations

import numpy as np
import pytest

from perception.capture import Screenshot
from perception.element_detector import ElementDetector
from perception.ocr_engine import OCREngine, OCRResult, PaddleOCREngine
from perception.preprocess import AGGRESSIVE, DEFAULT, PASSTHROUGH, PreprocessConfig, preprocess
from perception.types import BBox, ElementSource, UIElement
from perception.uia_tree import UIATree, find_by_name
from perception.visualizer import draw_elements, draw_legend, save_annotated


def _canvas(h: int = 100, w: int = 200) -> np.ndarray:
    image = np.full((h, w, 3), 200, dtype=np.uint8)
    image[20:40, 30:120] = 40  # 一块深色，让 CLAHE / 二值化有东西可做
    return image


def _shot(region: BBox) -> Screenshot:
    return Screenshot(
        image=np.full((region.height, region.width, 3), 128, dtype=np.uint8),
        region=region,
        engine="fake",
    )


# ===================================================================== #
# 预处理链
# ===================================================================== #


def test_passthrough_does_not_change_image() -> None:
    """基线组必须是真正的恒等变换——它是双引擎对照实验的对照组。"""
    image = _canvas()
    assert np.array_equal(preprocess(image, PASSTHROUGH), image)


def test_preprocess_does_not_mutate_input() -> None:
    image = _canvas()
    original = image.copy()
    preprocess(image, AGGRESSIVE)
    assert np.array_equal(image, original)


def test_grayscale_reduces_channels() -> None:
    result = preprocess(_canvas(), PreprocessConfig(grayscale=True, clahe=False, upscale=1.0))
    assert result.ndim == 2


def test_upscale_changes_size() -> None:
    result = preprocess(_canvas(100, 200), PreprocessConfig(grayscale=False, clahe=False, upscale=2.0))
    assert result.shape[:2] == (200, 400)


def test_binarize_produces_two_values() -> None:
    result = preprocess(_canvas(), AGGRESSIVE)
    assert set(np.unique(result)).issubset({0, 255})


def test_clahe_on_color_preserves_channels() -> None:
    """彩色图的 CLAHE 走 LAB 空间只动亮度，通道数不能变、也不该产生色偏。"""
    result = preprocess(_canvas(), PreprocessConfig(grayscale=False, clahe=True, upscale=1.0))
    assert result.shape[2] == 3


def test_default_does_not_binarize() -> None:
    """默认不二值化——深度学习 OCR 吃二值图往往更差，见 preprocess 模块文档。"""
    assert DEFAULT.binarize is False
    assert "binarize" not in " ".join(DEFAULT.enabled_steps())


def test_enabled_steps_reports_actual_chain() -> None:
    steps = " ".join(AGGRESSIVE.enabled_steps())
    for expected in ("grayscale", "clahe", "upscale", "binarize", "denoise"):
        assert expected in steps


# ===================================================================== #
# OCR 引擎的胶水逻辑
# ===================================================================== #


class FakeOCREngine(OCREngine):
    """返回预设结果的假引擎。用来测基类的预处理、还原、过滤逻辑。"""

    name = "fake"

    def __init__(self, results: list[OCRResult], config: PreprocessConfig = DEFAULT) -> None:
        super().__init__(config=config)
        self._results = results
        self.received_shape: tuple | None = None

    def is_available(self) -> bool:
        return True

    def _raw_recognize(self, image: np.ndarray) -> list[OCRResult]:
        self.received_shape = image.shape
        return list(self._results)


def test_upscale_is_undone_in_coordinates() -> None:
    """2× 超分后坐标必须除回去。忘记还原是这类流水线最常见的坐标 bug。"""
    raw = [OCRResult("确定", BBox(100, 200, 300, 260), 0.95)]
    engine = FakeOCREngine(raw, PreprocessConfig(grayscale=False, clahe=False, upscale=2.0))

    result = engine.recognize(_canvas())[0]
    assert result.bbox.as_tuple() == (50, 100, 150, 130)


def test_no_rescale_when_upscale_is_one() -> None:
    raw = [OCRResult("确定", BBox(10, 20, 30, 40), 0.9)]
    engine = FakeOCREngine(raw, PASSTHROUGH)
    assert engine.recognize(_canvas())[0].bbox.as_tuple() == (10, 20, 30, 40)


def test_polygon_is_rescaled_too() -> None:
    raw = [OCRResult("x", BBox(10, 10, 20, 20), 0.9, polygon=[(10, 10), (20, 10), (20, 20), (10, 20)])]
    engine = FakeOCREngine(raw, PreprocessConfig(grayscale=False, clahe=False, upscale=2.0))
    assert engine.recognize(_canvas())[0].polygon[0] == (5, 5)


def test_low_confidence_results_filtered() -> None:
    raw = [
        OCRResult("好", BBox(0, 0, 10, 10), 0.9),
        OCRResult("差", BBox(20, 0, 30, 10), 0.2),
    ]
    engine = FakeOCREngine(raw, PASSTHROUGH)
    engine.min_confidence = 0.5
    assert [r.text for r in engine.recognize(_canvas())] == ["好"]


def test_blank_text_filtered() -> None:
    raw = [OCRResult("   ", BBox(0, 0, 10, 10), 0.99), OCRResult("有内容", BBox(0, 20, 10, 30), 0.99)]
    engine = FakeOCREngine(raw, PASSTHROUGH)
    assert len(engine.recognize(_canvas())) == 1


def test_benchmark_is_recorded() -> None:
    engine = FakeOCREngine([OCRResult("a", BBox(0, 0, 5, 5), 0.8)], PASSTHROUGH)
    engine.recognize(_canvas())
    benchmark = engine.last_benchmark
    assert benchmark.engine == "fake"
    assert benchmark.num_results == 1
    assert benchmark.mean_confidence == pytest.approx(0.8)
    assert benchmark.elapsed_ms >= 0


def test_engine_receives_preprocessed_image() -> None:
    engine = FakeOCREngine([], PreprocessConfig(grayscale=True, clahe=False, upscale=2.0))
    engine.recognize(_canvas(100, 200))
    assert engine.received_shape == (200, 400)  # 灰度 + 2× 超分


def test_bbox_from_polygon() -> None:
    bbox, polygon = OCREngine._bbox_from_polygon([[10, 20], [50, 22], [48, 60], [12, 58]])
    assert bbox == BBox(10, 20, 51, 61)
    assert polygon[0] == (10, 20)


# --- PaddleOCR 输出格式解析（版本间变过多次，必须兜住）--- #


def test_parse_paddle_v2_format() -> None:
    raw = [[[[[0, 0], [10, 0], [10, 8], [0, 8]], ("确定", 0.97)]]]
    results = PaddleOCREngine._parse(raw)
    assert len(results) == 1
    assert results[0].text == "确定"
    assert results[0].confidence == pytest.approx(0.97)


def test_parse_handles_none_result() -> None:
    """无识别结果时某些版本返回 [None]，不能因此崩掉。"""
    assert PaddleOCREngine._parse([None]) == []


@pytest.mark.parametrize("raw", [[], None, [[]]])
def test_parse_handles_empty(raw) -> None:
    assert PaddleOCREngine._parse(raw) == []


def test_parse_skips_malformed_lines_but_keeps_good_ones() -> None:
    """一条坏数据不该让整张图的识别结果全丢。"""
    raw = [[
        [[[0, 0], [10, 0], [10, 8], [0, 8]], ("好", 0.9)],
        ["坏数据"],
        [[[0, 20], [10, 20], [10, 28], [0, 28]], ("也好", 0.9)],
    ]]
    assert [r.text for r in PaddleOCREngine._parse(raw)] == ["好", "也好"]


# ===================================================================== #
# 双通道融合 —— 坐标偏移是这里的头号雷
# ===================================================================== #


class FakeUIATree(UIATree):
    def __init__(self, elements: list[UIElement]) -> None:
        super().__init__()
        self._elements = elements

    @staticmethod
    def is_available() -> bool:
        return True

    def capture_foreground(self, clip: BBox | None = None) -> list[UIElement]:
        return list(self._elements)

    def capture_desktop(self, clip: BBox | None = None) -> list[UIElement]:
        return list(self._elements)


def test_ocr_coordinates_are_offset_to_screen_space() -> None:
    """OCR 给的是图像相对坐标，UIA 给的是屏幕绝对坐标，融合前必须统一。

    截全屏时两者恰好重合，这个 bug 在开发期不会暴露；一旦改成截区域或
    副显示器就会整体偏移。这条测试专门锁住它。
    """
    region = BBox(500, 300, 1300, 900)  # 非零原点
    ocr = FakeOCREngine([OCRResult("按钮", BBox(10, 20, 60, 45), 0.95)], PASSTHROUGH)
    detector = ElementDetector(ocr_engine=ocr, uia_tree=FakeUIATree([]), parallel=False)

    element = detector.detect(_shot(region), use_uia=False).elements[0]
    assert element.bbox.as_tuple() == (510, 320, 560, 345)


def test_uia_coordinates_are_not_offset() -> None:
    """UIA 本来就是屏幕坐标，再加一次偏移就错了。"""
    region = BBox(500, 300, 1300, 900)
    uia_element = UIElement(bbox=BBox(600, 400, 700, 440), source=ElementSource.UIA, text="保存")
    detector = ElementDetector(
        ocr_engine=None, uia_tree=FakeUIATree([uia_element]), parallel=False
    )

    element = detector.detect(_shot(region), use_ocr=False).elements[0]
    assert element.bbox.as_tuple() == (600, 400, 700, 440)


def test_uia_wins_over_ocr_in_fusion() -> None:
    region = BBox(0, 0, 800, 600)
    uia_element = UIElement(bbox=BBox(100, 100, 300, 140), source=ElementSource.UIA, text="保存")
    ocr = FakeOCREngine([OCRResult("保存", BBox(150, 110, 250, 130), 0.95)], PASSTHROUGH)
    detector = ElementDetector(
        ocr_engine=ocr, uia_tree=FakeUIATree([uia_element]), iou_threshold=0.1, parallel=False
    )

    result = detector.detect(_shot(region))
    assert result.uia_count == 1
    assert result.ocr_count == 1
    assert result.fused_count == 1
    assert result.dropped_by_dedupe == 1
    assert result.elements[0].source is ElementSource.UIA


def test_detection_summary_has_per_source_counts() -> None:
    """M1 验收要求按来源分项统计召回率，依赖这些计数。"""
    region = BBox(0, 0, 800, 600)
    ocr = FakeOCREngine([OCRResult("a", BBox(0, 0, 20, 20), 0.9)], PASSTHROUGH)
    uia_element = UIElement(bbox=BBox(400, 400, 500, 440), source=ElementSource.UIA, text="b")
    detector = ElementDetector(
        ocr_engine=ocr, uia_tree=FakeUIATree([uia_element]), parallel=False
    )

    result = detector.detect(_shot(region))
    summary = result.summary()
    assert summary["uia_raw"] == 1
    assert summary["ocr_raw"] == 1
    assert summary["fused"] == 2
    assert len(result.by_source(ElementSource.UIA)) == 1
    assert len(result.by_source(ElementSource.OCR)) == 1


def test_ocr_failure_does_not_kill_detection() -> None:
    """一个通道挂掉时另一个必须继续工作——感知不能因为 OCR 崩了就全瞎。"""

    class ExplodingEngine(FakeOCREngine):
        def _raw_recognize(self, image):
            raise RuntimeError("OCR 炸了")

    region = BBox(0, 0, 800, 600)
    uia_element = UIElement(bbox=BBox(10, 10, 100, 50), source=ElementSource.UIA, text="仍在")
    detector = ElementDetector(
        ocr_engine=ExplodingEngine([], PASSTHROUGH),
        uia_tree=FakeUIATree([uia_element]),
        parallel=False,
    )

    result = detector.detect(_shot(region))
    assert result.fused_count == 1
    assert result.ocr_count == 0


def test_parallel_and_serial_give_same_result() -> None:
    """并行只是性能优化，不能改变结果。"""
    region = BBox(100, 100, 900, 700)
    uia_element = UIElement(bbox=BBox(400, 400, 500, 440), source=ElementSource.UIA, text="b")

    outputs = []
    for parallel in (False, True):
        ocr = FakeOCREngine([OCRResult("a", BBox(0, 0, 20, 20), 0.9)], PASSTHROUGH)
        detector = ElementDetector(
            ocr_engine=ocr, uia_tree=FakeUIATree([uia_element]), parallel=parallel
        )
        outputs.append([e.bbox.as_tuple() for e in detector.detect(_shot(region)).elements])
    assert outputs[0] == outputs[1]


# ===================================================================== #
# UIA 过滤逻辑
# ===================================================================== #


class FakeRect:
    def __init__(self, left, top, right, bottom):
        self.left, self.top, self.right, self.bottom = left, top, right, bottom

    def width(self):
        return self.right - self.left

    def height(self):
        return self.bottom - self.top


class FakeNode:
    def __init__(self, name="", control_type="ButtonControl", rect=(0, 0, 100, 40), offscreen=False):
        self.Name = name
        self.ControlTypeName = control_type
        self.BoundingRectangle = FakeRect(*rect) if rect else None
        self.IsOffscreen = offscreen
        self.AutomationId = "id-1"
        self.ClassName = "Cls"
        self.IsEnabled = True


@pytest.fixture
def tree() -> UIATree:
    return UIATree()


def test_offscreen_node_rejected(tree: UIATree) -> None:
    assert tree._to_element(FakeNode(name="隐藏", offscreen=True), None) is None


def test_zero_rect_node_rejected(tree: UIATree) -> None:
    """未布局的控件会给出零矩形，留着会污染结果并产生无效点击目标。"""
    assert tree._to_element(FakeNode(name="未布局", rect=(10, 10, 10, 10)), None) is None


def test_tiny_node_rejected(tree: UIATree) -> None:
    assert tree._to_element(FakeNode(name="小", rect=(0, 0, 4, 4)), None) is None


def test_container_rejected_by_default(tree: UIATree) -> None:
    assert tree._to_element(FakeNode(name="面板", control_type="PaneControl"), None) is None


def test_container_kept_when_requested() -> None:
    tree = UIATree(include_containers=True)
    assert tree._to_element(FakeNode(name="面板", control_type="PaneControl"), None) is not None


def test_unnamed_non_interactive_rejected(tree: UIATree) -> None:
    assert tree._to_element(FakeNode(name="", control_type="TextControl"), None) is None


def test_unnamed_interactive_kept(tree: UIATree) -> None:
    """无名但可交互的控件要留——图标按钮通常没有 Name。"""
    assert tree._to_element(FakeNode(name="", control_type="ButtonControl"), None) is not None


def test_node_outside_clip_rejected(tree: UIATree) -> None:
    assert tree._to_element(FakeNode(name="远处", rect=(5000, 5000, 5100, 5040)), BBox(0, 0, 800, 600)) is None


def test_valid_node_carries_metadata(tree: UIATree) -> None:
    element = tree._to_element(FakeNode(name="保存"), None)
    assert element.source is ElementSource.UIA
    assert element.text == "保存"
    assert element.meta["automation_id"] == "id-1"


def test_broken_node_does_not_raise(tree: UIATree) -> None:
    """跨进程 COM 调用什么都可能抛，一个坏控件不该中断整棵树的遍历。"""

    class Broken:
        @property
        def IsOffscreen(self):
            raise OSError("COM 调用失败")

    assert tree._to_element(Broken(), None) is None


def test_find_by_name() -> None:
    elements = [
        UIElement(bbox=BBox(0, 0, 10, 10), source=ElementSource.UIA, text="文件"),
        UIElement(bbox=BBox(0, 20, 10, 30), source=ElementSource.UIA, text="编辑"),
    ]
    assert find_by_name(elements, "编辑").text == "编辑"
    assert find_by_name(elements, "编") is not None
    assert find_by_name(elements, "编", exact=True) is None
    assert find_by_name(elements, "不存在") is None


# ===================================================================== #
# 可视化
# ===================================================================== #


def _elements() -> list[UIElement]:
    return [
        UIElement(bbox=BBox(10, 10, 110, 50), source=ElementSource.UIA, text="保存", index=1),
        UIElement(bbox=BBox(10, 60, 160, 90), source=ElementSource.OCR, text="用户名", index=2),
    ]


def test_draw_elements_preserves_shape() -> None:
    image = _canvas(200, 300)
    assert draw_elements(image, _elements()).shape == image.shape


def test_draw_elements_actually_draws() -> None:
    image = _canvas(200, 300)
    assert not np.array_equal(draw_elements(image, _elements()), image)


def test_draw_elements_applies_origin_offset() -> None:
    """元素坐标是屏幕绝对坐标，画到区域截图上必须减掉区域原点。

    传错 origin 时框会整体偏移——两种画法结果必须不同，否则说明 origin
    根本没被用上。
    """
    image = _canvas(200, 300)
    without = draw_elements(image, _elements(), origin=(0, 0))
    with_offset = draw_elements(image, _elements(), origin=(5, 5))
    assert not np.array_equal(without, with_offset)


def test_draw_elements_handles_empty_list() -> None:
    image = _canvas()
    assert np.array_equal(draw_elements(image, []), image)


def test_label_at_top_edge_does_not_crash() -> None:
    """贴着屏幕顶部的元素（菜单栏）标签没有向上的空间，必须改画到框内。"""
    image = _canvas(200, 300)
    top_element = [UIElement(bbox=BBox(0, 0, 100, 20), source=ElementSource.UIA, text="文件", index=1)]
    assert draw_elements(image, top_element).shape == image.shape


def test_draw_legend_adds_content() -> None:
    image = _canvas(300, 400)
    result = draw_legend(image, {"uia_raw": 5, "ocr_raw": 3, "fused": 6, "total_ms": 42.0})
    assert not np.array_equal(result, image)


def test_save_annotated_writes_file(tmp_path) -> None:
    path = tmp_path / "sub" / "shot.png"
    returned = save_annotated(_canvas(200, 300), _elements(), str(path))
    assert path.exists() and path.stat().st_size > 0
    assert returned == str(path)


def test_save_annotated_handles_chinese_path(tmp_path) -> None:
    """cv2.imwrite 在中文路径下会静默失败，实现里绕开了它——这条锁住该行为。"""
    path = tmp_path / "中文目录" / "截图.png"
    save_annotated(_canvas(), _elements(), str(path))
    assert path.exists() and path.stat().st_size > 0
