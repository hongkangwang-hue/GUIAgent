"""截图模块测试。

标记为 `windows` 的用例需要真实桌面，CI 上用 `-m "not windows"` 跳过。
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from perception.capture import CaptureError, ScreenCapturer, Screenshot
from perception.types import BBox

#: 两个标记都要打，作用不同，缺一个就会出问题：
#: - `skipif` 让本地在非 Windows 上自动跳过
#: - `windows` 是自定义标记，供 CI 用 `-m "not windows"` 主动排除
#: 只打 skipif 的话，CI 的过滤是一句空话——这正是最初的写法，230 个测试
#: 一个都没被排除掉。
windows_only = pytest.mark.windows(
    pytest.mark.skipif(sys.platform != "win32", reason="需要 Windows 桌面")
)


# --------------------------------------------------------------------- #
# Screenshot 数据类（不需要真实屏幕）
# --------------------------------------------------------------------- #


def _fake_shot(w: int = 200, h: int = 100) -> Screenshot:
    return Screenshot(
        image=np.zeros((h, w, 3), dtype=np.uint8),
        region=BBox(0, 0, w, h),
        engine="fake",
    )


def test_screenshot_dimensions_match_image() -> None:
    shot = _fake_shot(320, 240)
    assert (shot.width, shot.height) == (320, 240)


def test_resize_to_produces_requested_size() -> None:
    resized = _fake_shot(2560, 1600).resize_to(1024, 768)
    assert resized.shape[:2] == (768, 1024)


def test_to_pil_roundtrip_preserves_size() -> None:
    shot = _fake_shot(64, 48)
    assert shot.to_pil().size == (64, 48)


# --------------------------------------------------------------------- #
# 真实屏幕
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def capturer():
    try:
        with ScreenCapturer() as cap:
            yield cap
    except CaptureError as exc:
        pytest.skip(f"无可用截图引擎：{exc}")


@windows_only
def test_engine_is_selected(capturer: ScreenCapturer) -> None:
    assert capturer.engine_name in {"dxcam", "mss", "pyautogui"}


@windows_only
def test_capture_returns_bgr_image(capturer: ScreenCapturer) -> None:
    shot = capturer.capture()
    assert shot.image.ndim == 3
    assert shot.image.shape[2] == 3
    assert shot.image.dtype == np.uint8


@windows_only
def test_capture_matches_monitor_region(capturer: ScreenCapturer) -> None:
    """图像尺寸必须与声明的区域一致——不一致时坐标换算全盘皆错。"""
    region = capturer.monitor_region(1)
    shot = capturer.capture_region(region)
    assert (shot.width, shot.height) == (region.width, region.height)


@windows_only
def test_capture_subregion(capturer: ScreenCapturer) -> None:
    region = BBox(100, 100, 500, 400)
    shot = capturer.capture_region(region)
    assert (shot.width, shot.height) == (400, 300)
    assert shot.region == region


@windows_only
def test_capture_is_dpi_physical_pixels(capturer: ScreenCapturer) -> None:
    """在非 100% 缩放下，截图必须是物理像素而非逻辑像素。

    这条挂掉说明 DPI 感知没生效，此时所有点击都会系统性偏移。
    """
    import ctypes

    region = capturer.monitor_region(1)
    physical_width = ctypes.windll.user32.GetSystemMetrics(0)
    assert abs(region.width - physical_width) <= 1


@windows_only
def test_reject_degenerate_region(capturer: ScreenCapturer) -> None:
    with pytest.raises(ValueError):
        capturer.capture_region(BBox(10, 10, 10, 20))


@windows_only
def test_monitor_index_out_of_range(capturer: ScreenCapturer) -> None:
    with pytest.raises(IndexError):
        capturer.monitor_region(99)


@windows_only
@pytest.mark.slow
def test_latency_meets_acceptance_criterion(capturer: ScreenCapturer) -> None:
    """M1 验收标准：单帧截图延迟低于 15ms。

    看 p95 不看均值——偶发慢帧会打乱 Agent 的动作时序，均值会把它们平均掉。
    """
    stats = capturer.benchmark(n=50)
    assert stats["p95_ms"] < 15.0, f"截图 p95 {stats['p95_ms']}ms 超出 15ms 预算：{stats}"


@windows_only
@pytest.mark.slow
def test_fresh_mode_is_slower_but_bounded(capturer: ScreenCapturer) -> None:
    """`fresh=True` 会等一帧新画面，下限是显示器刷新周期。

    它比默认模式慢是**预期行为**，不是性能问题：动作发出后界面正在重绘，
    不等新帧就会拿到动作前的画面。
    """
    if capturer.engine_name != "dxcam":
        pytest.skip("只有 dxcam 区分 fresh / latest")
    fresh = capturer.benchmark(n=20, fresh=True)
    assert fresh["p95_ms"] < 60.0  # 宽松上限：即便 20Hz 刷新也该在此之内
