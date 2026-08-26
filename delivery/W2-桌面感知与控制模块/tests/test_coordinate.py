"""坐标系统测试。

坐标是 M1 的地基，因此这里要求接近全覆盖。本文件**只依赖标准库 + pytest**，
在没装 mss / OpenCV 的机器上也能跑。
"""

from __future__ import annotations

import pytest

from perception.coordinate import CoordinateScaler, ScaleMode
from perception.types import BBox, Point

FHD = BBox(0, 0, 1920, 1080)
QHD_PLUS = BBox(0, 0, 2560, 1600)  # 本机 ROG 幻16 Air 的分辨率


@pytest.fixture
def scaler() -> CoordinateScaler:
    s = CoordinateScaler(QHD_PLUS)
    s.register("planner", 1024, 768)
    s.register("grounding", 1280, 800)
    s.register("normalized", 1000, 1000)
    return s


# --------------------------------------------------------------------- #
# 注册与查询
# --------------------------------------------------------------------- #


def test_register_and_get(scaler: CoordinateScaler) -> None:
    space = scaler.get("planner")
    assert (space.width, space.height) == (1024, 768)
    assert space.mode is ScaleMode.STRETCH


def test_get_unknown_space_lists_known_ones(scaler: CoordinateScaler) -> None:
    with pytest.raises(KeyError) as exc:
        scaler.get("nope")
    # 报错信息要能自解释，否则调试时还得翻代码找注册在哪
    assert "planner" in str(exc.value)


def test_multiple_named_spaces_coexist(scaler: CoordinateScaler) -> None:
    """验收标准 2：至少两个命名坐标系可注册并互转。"""
    assert set(scaler.spaces) == {"planner", "grounding", "normalized"}


def test_reject_degenerate_space(scaler: CoordinateScaler) -> None:
    with pytest.raises(ValueError):
        scaler.register("bad", 0, 100)


def test_reject_degenerate_region() -> None:
    with pytest.raises(ValueError):
        CoordinateScaler(BBox(10, 10, 10, 20))


# --------------------------------------------------------------------- #
# 往返一致性 —— 验收标准 1
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("space", ["planner", "grounding", "normalized"])
def test_roundtrip_within_declared_bound(scaler: CoordinateScaler, space: str) -> None:
    """真实 → 模型 → 真实 的误差不超过该坐标系自己声明的上界。

    对照 `roundtrip_error_bound()` 而非写死常数：换分辨率时常数会悄悄失效，
    而上界是从缩放比算出来的，永远跟着实际配置走。
    """
    bound = scaler.roundtrip_error_bound(space)
    step = 37  # 取质数步长，避免只测到恰好整除的"好"坐标
    for x in range(0, QHD_PLUS.width, step):
        for y in range(0, QHD_PLUS.height, step):
            original = Point(x, y)
            back = scaler.to_real(scaler.to_model(original, space), space)
            assert abs(back.x - original.x) <= bound + 1
            assert abs(back.y - original.y) <= bound + 1


def test_planner_space_meets_2px_acceptance(scaler: CoordinateScaler) -> None:
    """验收标准 1 的字面要求：往返误差不超过 2px。"""
    assert scaler.roundtrip_error_bound("planner") <= 2.0
    assert scaler.roundtrip_error_bound("grounding") <= 2.0


def test_error_bound_grows_with_aggressive_downscale() -> None:
    """降分辨率过猛时上界会突破 2px —— 这是设计上要暴露的事实，不是 bug。"""
    s = CoordinateScaler(BBox(0, 0, 3840, 2160))
    s.register("tiny", 640, 360)
    assert s.roundtrip_error_bound("tiny") > 2.0


def test_model_to_real_to_model_is_stable(scaler: CoordinateScaler) -> None:
    """模型 → 真实 → 模型 必须完全一致（模型空间更粗，不该有信息损失）。"""
    for x in range(0, 1024, 13):
        for y in range(0, 768, 11):
            p = Point(x, y)
            assert scaler.to_model(scaler.to_real(p, "planner"), "planner") == p


# --------------------------------------------------------------------- #
# 边角与夹取
# --------------------------------------------------------------------- #


def test_origin_maps_to_origin(scaler: CoordinateScaler) -> None:
    assert scaler.to_model(Point(0, 0), "planner") == Point(0, 0)


def test_bottom_right_maps_inside_canvas(scaler: CoordinateScaler) -> None:
    p = scaler.to_model(Point(2559, 1599), "planner")
    assert p == Point(1023, 767)


def test_out_of_range_model_point_is_clamped(scaler: CoordinateScaler) -> None:
    """模型幻觉出画布外的坐标时夹到边界，而不是抛异常或产生负数屏幕坐标。"""
    real = scaler.to_real(Point(99999, -50), "planner")
    assert scaler.region.contains(real)


def test_out_of_range_real_point_is_clamped(scaler: CoordinateScaler) -> None:
    model = scaler.to_model(Point(-100, 99999), "planner")
    assert 0 <= model.x < 1024
    assert 0 <= model.y < 768


# --------------------------------------------------------------------- #
# 多显示器偏移
# --------------------------------------------------------------------- #


def test_secondary_monitor_offset_is_applied() -> None:
    """副显示器在虚拟桌面上有偏移，模型坐标必须相对该显示器而非虚拟桌面原点。"""
    secondary = BBox.from_xywh(2560, 0, 1920, 1080)
    s = CoordinateScaler(secondary)
    s.register("planner", 1024, 768)

    assert s.to_model(Point(2560, 0), "planner") == Point(0, 0)

    center_real = s.to_real(Point(512, 384), "planner")
    assert 2560 <= center_real.x < 4480
    assert s.region.contains(center_real)


def test_from_monitor_dict() -> None:
    s = CoordinateScaler.from_monitor({"left": 100, "top": 50, "width": 800, "height": 600})
    assert s.region == BBox(100, 50, 900, 650)


def test_is_in_region() -> None:
    s = CoordinateScaler(BBox.from_xywh(2560, 0, 1920, 1080))
    assert s.is_in_region(Point(3000, 500))
    assert not s.is_in_region(Point(100, 500))


# --------------------------------------------------------------------- #
# 坐标系互转 —— 模式 B 依赖
# --------------------------------------------------------------------- #


def test_convert_between_spaces(scaler: CoordinateScaler) -> None:
    """规划模型在 planner 空间给点，grounding 模型工作在另一个空间。"""
    planner_center = Point(512, 384)
    grounding_point = scaler.convert(planner_center, "planner", "grounding")
    assert abs(grounding_point.x - 640) <= 2
    assert abs(grounding_point.y - 400) <= 2


def test_convert_same_space_is_identity(scaler: CoordinateScaler) -> None:
    p = Point(123, 456)
    assert scaler.convert(p, "planner", "planner") is p


def test_normalized_space_matches_qwen_convention(scaler: CoordinateScaler) -> None:
    """Qwen2.5-VL 一类模型输出 0-1000 归一化坐标，注册成 1000×1000 即可直接用。"""
    real = scaler.to_real(Point(500, 500), "normalized")
    assert abs(real.x - 1280) <= 3
    assert abs(real.y - 800) <= 3


# --------------------------------------------------------------------- #
# FIT 模式
# --------------------------------------------------------------------- #


def test_fit_mode_preserves_aspect_ratio() -> None:
    """16:10 的屏幕放进 1:1 画布，上下应留边，且往返仍然正确。"""
    s = CoordinateScaler(QHD_PLUS)
    s.register("square", 1000, 1000, mode=ScaleMode.FIT)

    top_left = s.to_model(Point(0, 0), "square")
    assert top_left.x == 0
    assert top_left.y > 0  # 上方有黑边

    bound = s.roundtrip_error_bound("square")
    for x in range(0, 2560, 97):
        for y in range(0, 1600, 61):
            back = s.to_real(s.to_model(Point(x, y), "square"), "square")
            assert abs(back.x - x) <= bound + 1
            assert abs(back.y - y) <= bound + 1


def test_stretch_mode_fills_canvas() -> None:
    s = CoordinateScaler(QHD_PLUS)
    s.register("square", 1000, 1000, mode=ScaleMode.STRETCH)
    assert s.to_model(Point(0, 0), "square") == Point(0, 0)
    assert s.to_model(Point(2559, 1599), "square") == Point(999, 999)


# --------------------------------------------------------------------- #
# 边界框转换
# --------------------------------------------------------------------- #


def test_bbox_roundtrip(scaler: CoordinateScaler) -> None:
    original = BBox(100, 200, 500, 400)
    back = scaler.bbox_to_real(scaler.bbox_to_model(original, "planner"), "planner")
    for a, b in zip(original.as_tuple(), back.as_tuple(), strict=True):
        assert abs(a - b) <= 3


def test_bbox_stays_valid_after_scaling(scaler: CoordinateScaler) -> None:
    """极小的框缩放后不能变成非法框（right < left）。"""
    tiny = BBox(1000, 1000, 1001, 1001)
    scaled = scaler.bbox_to_model(tiny, "planner")
    assert scaled.right >= scaled.left
    assert scaled.bottom >= scaled.top


def test_bbox_center_survives_roundtrip(scaler: CoordinateScaler) -> None:
    """点击点取的是框中心，中心的往返精度才是真正要保证的。"""
    original = BBox(300, 400, 700, 600)
    model_box = scaler.bbox_to_model(original, "planner")
    real_center = scaler.to_real(model_box.center, "planner")
    assert abs(real_center.x - original.center.x) <= 3
    assert abs(real_center.y - original.center.y) <= 3


# --------------------------------------------------------------------- #
# DPI 缩放场景 —— 验收标准 3
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("scale", [1.0, 1.25, 1.5])
def test_click_target_correct_under_dpi_scaling(scale: float) -> None:
    """三档 DPI 缩放下点击目标位置均正确。

    声明 DPI 感知后截图拿到的是物理像素，缩放比例只改变屏幕的物理分辨率，
    而 `CoordinateScaler` 以实际截图区域为准——因此换算不受缩放影响。
    这个测试锁住的正是这个性质：**任何让缩放比例泄漏进坐标运算的改动都会
    让它挂掉。**
    """
    physical_w, physical_h = 1920, 1080
    logical_w = int(physical_w / scale)
    logical_h = int(physical_h / scale)

    s = CoordinateScaler(BBox(0, 0, physical_w, physical_h))
    s.register("planner", 1024, 768)

    # 逻辑坐标系里的一个按钮中心，换算成物理像素
    button_logical = Point(logical_w // 2, logical_h // 2)
    button_physical = Point(int(button_logical.x * scale), int(button_logical.y * scale))

    model_point = s.to_model(button_physical, "planner")
    clicked = s.to_real(model_point, "planner")

    assert abs(clicked.x - button_physical.x) <= 2
    assert abs(clicked.y - button_physical.y) <= 2


# --------------------------------------------------------------------- #


def test_repr_is_informative(scaler: CoordinateScaler) -> None:
    text = repr(scaler)
    assert "planner(1024×768)" in text
