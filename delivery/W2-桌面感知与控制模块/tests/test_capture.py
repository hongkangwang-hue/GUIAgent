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
def test_fresh_mode_is_bounded_by_its_timeout(capturer: ScreenCapturer) -> None:
    """`fresh=True` 等一帧新画面，但**绝不会无限等下去**。

    这里断言的是上界语义，不是某个性能数字。原先写的是 `p95 < 60ms`，
    那条断言构造上就不可靠——「等一帧新画面」要多久**完全取决于屏幕当时
    有没有在动**：桌面静止时根本没有新帧，必然等满 `FRESH_TIMEOUT_S`
    然后复用缓存，p95 就是 500ms。在无人操作的机器上跑测试正是这种情况。

    真正该守住的不变量只有两条：等待有上界，且拿得到一张可用的画面。
    至于「有多快」，那是环境属性（显示器 / 虚拟显卡的出帧率），不该由
    单元测试来判定——验收标准 1 用的也不是这个口径，而是 `fresh=False`
    的「取当前画面」（见 `scripts/env_check.py`）。
    """
    if capturer.engine_name != "dxcam":
        pytest.skip("只有 dxcam 区分 fresh / latest")

    timeout_ms = capturer._engine.FRESH_TIMEOUT_S * 1000.0
    fresh = capturer.benchmark(n=10, fresh=True)

    # 留 30% 余量给调度抖动；关键是它有界，而不是界在哪
    assert fresh["max_ms"] < timeout_ms * 1.3, (
        f"fresh 模式最长 {fresh['max_ms']}ms，超过了 {timeout_ms}ms 的超时上界：{fresh}"
    )

    shot = capturer.capture(fresh=True)
    assert shot.width > 0 and shot.height > 0
