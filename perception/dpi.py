"""Windows DPI 感知。

**必须在任何截图或坐标查询之前调用 `enable_dpi_awareness()`。**

不声明 DPI 感知时，Windows 会对进程撒谎：在 150% 缩放的 2560×1600 屏幕上，
`GetSystemMetrics` 返回的是 1707×1067（逻辑像素），而 mss 截到的是 2560×1600
（物理像素）。两者不一致，点击就会系统性偏移——而且偏移量随缩放比例变化，
在 100% 的机器上测不出来。这是 GUI 自动化最典型的坑。

本模块在非 Windows 平台上全部退化为安全的空操作，便于在 CI 上跑单元测试。
"""

from __future__ import annotations

import ctypes
import logging
import sys

logger = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"

# DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2，Win10 1703+
_DPI_CONTEXT_PER_MONITOR_V2 = ctypes.c_void_p(-4)
# PROCESS_PER_MONITOR_DPI_AWARE，Win8.1+
_PROCESS_PER_MONITOR_DPI_AWARE = 2
# 已经设置过时 SetProcessDpiAwareness 返回 E_ACCESSDENIED，这不是错误
_E_ACCESSDENIED = -2147024891

#: Windows 的 DPI 基准值。96 DPI = 100% 缩放。
BASE_DPI = 96.0

_awareness_enabled = False


def enable_dpi_awareness() -> bool:
    """声明本进程为 per-monitor DPI 感知。

    幂等：重复调用安全。返回是否处于 DPI 感知状态。

    按新到旧依次尝试三种 API，因为它们的可用性跨 Windows 版本不同：
    1. ``SetProcessDpiAwarenessContext``（Win10 1703+，支持每显示器 V2）
    2. ``SetProcessDpiAwareness``（Win8.1+）
    3. ``SetProcessDPIAware``（Vista+，只有系统级感知）
    """
    global _awareness_enabled

    if not IS_WINDOWS:
        logger.debug("非 Windows 平台，跳过 DPI 感知设置")
        return False
    if _awareness_enabled:
        return True

    try:
        if ctypes.windll.user32.SetProcessDpiAwarenessContext(_DPI_CONTEXT_PER_MONITOR_V2):
            _awareness_enabled = True
            logger.info("DPI 感知已启用：per-monitor V2")
            return True
    except (AttributeError, OSError):
        pass

    try:
        hresult = ctypes.windll.shcore.SetProcessDpiAwareness(_PROCESS_PER_MONITOR_DPI_AWARE)
        if hresult == 0 or hresult == _E_ACCESSDENIED:
            _awareness_enabled = True
            logger.info("DPI 感知已启用：per-monitor")
            return True
    except (AttributeError, OSError):
        pass

    try:
        ctypes.windll.user32.SetProcessDPIAware()
        _awareness_enabled = True
        logger.warning("DPI 感知已启用：仅系统级。多显示器不同缩放时坐标可能偏移")
        return True
    except (AttributeError, OSError) as exc:
        logger.error("DPI 感知设置失败：%s。坐标将不可靠", exc)
        return False


def is_dpi_aware() -> bool:
    """本进程当前是否为 DPI 感知状态。"""
    return _awareness_enabled


def get_system_dpi() -> int:
    """系统 DPI。100% 缩放为 96。"""
    if not IS_WINDOWS:
        return int(BASE_DPI)
    try:
        return int(ctypes.windll.user32.GetDpiForSystem())
    except (AttributeError, OSError):
        # Win10 1607 以前没有 GetDpiForSystem，退回从桌面 DC 读取
        try:
            hdc = ctypes.windll.user32.GetDC(0)
            dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX
            ctypes.windll.user32.ReleaseDC(0, hdc)
            return int(dpi)
        except (AttributeError, OSError):
            return int(BASE_DPI)


def get_scale_factor() -> float:
    """系统缩放比例。100% → 1.0，125% → 1.25，150% → 1.5。"""
    return get_system_dpi() / BASE_DPI


def describe() -> dict:
    """当前 DPI 状态摘要，供 ``env_check.py`` 与调试日志使用。"""
    return {
        "platform": sys.platform,
        "dpi_aware": _awareness_enabled,
        "system_dpi": get_system_dpi(),
        "scale_factor": get_scale_factor(),
    }
