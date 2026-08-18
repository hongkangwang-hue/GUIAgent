"""开发环境自检。

分两级输出：

- **必过项**：失败即阻塞 M1 验收，退出码 1
- **警告项**：仅提示，不影响退出码，**但 M3 开工前必须全部转绿**

GPU 相关全部是警告项。理由：M1/M2 阶段真正用到 GPU 的只有 PaddleOCR 的
推理加速，而它跑 CPU 完全可用（单张图从几十毫秒变几百毫秒）；规划层走
API，不占本地显存。把 GPU 设成阻塞项只会在第一周制造无谓的挫败感。

用法::

    python scripts/env_check.py
    python scripts/env_check.py --json      # 机器可读，CI 用
    python scripts/env_check.py --skip-ocr  # 跳过 OCR（首次加载要下模型）
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REQUIRED_PYTHON = (3, 10)

#: (模块名, pip 包名)。模块名与包名不一致的地方是踩坑高发区
REQUIRED_MODULES = [
    ("numpy", "numpy"),
    ("PIL", "pillow"),
    ("mss", "mss"),
    ("pyautogui", "pyautogui"),
    ("pynput", "pynput"),
    ("pyperclip", "pyperclip"),
    ("cv2", "opencv-python"),
]
WINDOWS_ONLY_MODULES = [("uiautomation", "uiautomation")]
OPTIONAL_MODULES = [
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("bitsandbytes", "bitsandbytes"),
    ("paddleocr", "paddleocr"),
    ("easyocr", "easyocr"),
    ("langchain", "langchain"),
]

SCREENSHOT_BUDGET_MS = 15.0


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""
    required: bool = True
    data: dict = field(default_factory=dict)


class Report:
    def __init__(self) -> None:
        self.checks: list[Check] = []

    def add(self, check: Check) -> Check:
        self.checks.append(check)
        return check

    @property
    def required_failures(self) -> list[Check]:
        return [c for c in self.checks if c.required and not c.passed]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if not c.required and not c.passed]

    def as_dict(self) -> dict:
        return {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "passed": not self.required_failures,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "required": c.required,
                    "detail": c.detail,
                    **c.data,
                }
                for c in self.checks
            ],
        }


# ====================================================================== #
# 必过项
# ====================================================================== #


def check_python(report: Report) -> None:
    version = sys.version_info[:2]
    ok = version >= REQUIRED_PYTHON
    report.add(
        Check(
            "Python 版本",
            ok,
            f"{sys.version.split()[0]}"
            + ("" if ok else f"（需要 ≥ {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}）"),
            data={"version": sys.version.split()[0]},
        )
    )


def check_dependencies(report: Report) -> None:
    modules = list(REQUIRED_MODULES)
    if sys.platform == "win32":
        modules += WINDOWS_ONLY_MODULES

    missing = []
    for module_name, package_name in modules:
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(package_name)

    report.add(
        Check(
            "核心依赖完整性",
            not missing,
            "全部就位" if not missing else f"缺失：pip install {' '.join(missing)}",
            data={"missing": missing},
        )
    )

    installed_optional = []
    for module_name, package_name in OPTIONAL_MODULES:
        try:
            importlib.import_module(module_name)
            installed_optional.append(package_name)
        except ImportError:
            pass
    report.add(
        Check(
            "后续里程碑依赖",
            True,  # 只报告，不判定
            f"已装 {len(installed_optional)}/{len(OPTIONAL_MODULES)}："
            + (", ".join(installed_optional) or "无"),
            required=False,
            data={"installed": installed_optional},
        )
    )


def check_screenshot_speed(report: Report) -> None:
    try:
        from perception.capture import ScreenCapturer

        with ScreenCapturer() as capturer:
            stats = capturer.benchmark(n=30)
    except Exception as exc:  # noqa: BLE001
        report.add(Check("截图速度", False, f"截图失败：{exc}"))
        return

    ok = stats["p95_ms"] < SCREENSHOT_BUDGET_MS
    report.add(
        Check(
            "截图速度",
            ok,
            f"{stats['engine']} @ {stats['resolution']}："
            f"p50 {stats['p50_ms']}ms / p95 {stats['p95_ms']}ms"
            + ("" if ok else f"（预算 {SCREENSHOT_BUDGET_MS}ms）"),
            data=stats,
        )
    )


def check_dpi(report: Report) -> None:
    try:
        from perception.dpi import describe, enable_dpi_awareness

        enable_dpi_awareness()
        info = describe()
    except Exception as exc:  # noqa: BLE001
        report.add(Check("DPI 缩放检测", False, str(exc)))
        return

    if sys.platform != "win32":
        report.add(Check("DPI 缩放检测", True, "非 Windows，跳过", data=info))
        return

    ok = info["dpi_aware"]
    report.add(
        Check(
            "DPI 缩放检测",
            ok,
            f"缩放 {info['scale_factor']:.0%}（{info['system_dpi']} DPI），"
            + ("已声明 DPI 感知" if ok else "未能声明 DPI 感知，坐标会系统性偏移"),
            data=info,
        )
    )


def check_ocr_cpu(report: Report, skip: bool = False) -> None:
    if skip:
        report.add(Check("OCR CPU 推理", True, "已按 --skip-ocr 跳过", required=False))
        return
    try:
        import numpy as np

        from perception.ocr_engine import PaddleOCREngine

        engine = PaddleOCREngine(use_gpu=False)
        if not engine.is_available():
            report.add(Check("OCR CPU 推理", False, "PaddleOCR 无法加载，检查 paddlepaddle 安装"))
            return

        canvas = np.full((100, 400, 3), 255, dtype=np.uint8)
        import cv2

        cv2.putText(canvas, "GUI Agent 2026", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

        start = time.perf_counter()
        results = engine.recognize(canvas)
        elapsed = (time.perf_counter() - start) * 1000.0

        report.add(
            Check(
                "OCR CPU 推理",
                True,
                f"识别 {len(results)} 条，耗时 {elapsed:.0f}ms"
                + (f"：{results[0].text!r}" if results else "（测试图未识别出内容，但引擎可用）"),
                data={"elapsed_ms": round(elapsed, 1), "num_results": len(results)},
            )
        )
    except ImportError as exc:
        report.add(Check("OCR CPU 推理", False, f"依赖缺失：{exc}"))
    except Exception as exc:  # noqa: BLE001
        report.add(Check("OCR CPU 推理", False, f"推理失败：{exc}"))


# ====================================================================== #
# 警告项 —— GPU 相关，M3 前必须转绿
# ====================================================================== #


def check_gpu(report: Report) -> None:
    try:
        import torch
    except ImportError:
        report.add(Check("CUDA 可用性", False, "torch 未安装（M3 前需要）", required=False))
        return

    available = torch.cuda.is_available()
    report.add(
        Check(
            "CUDA 可用性",
            available,
            f"torch {torch.__version__}" + ("" if available else "，CUDA 不可用"),
            required=False,
            data={"torch_version": torch.__version__},
        )
    )
    if not available:
        return

    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    arch_list = torch.get_arch_list()
    arch_tag = f"sm_{capability[0]}{capability[1]}"

    # 三重判据：算力标签、构建是否含该算力、以及真的跑一次运算。
    # is_available() 只检查驱动与运行时能否初始化，不检查有没有这张卡的
    # kernel —— 这正是 Blackwell 上最容易误判的地方。
    arch_supported = arch_tag in arch_list
    compute_ok = False
    compute_error = ""
    if arch_supported:
        try:
            x = torch.randn(512, 512, device="cuda")
            _ = (x @ x).sum().item()
            torch.cuda.synchronize()
            compute_ok = True
        except Exception as exc:  # noqa: BLE001
            compute_error = str(exc).split("\n")[0]

    ok = arch_supported and compute_ok
    if ok:
        detail = f"{name}（{arch_tag}），构建支持该算力，实测运算通过"
    elif not arch_supported:
        detail = (
            f"{name} 需要 {arch_tag}，但当前 torch 构建只含 {', '.join(arch_list)}。"
            f"改装 cu128+：pip install torch --index-url https://download.pytorch.org/whl/cu128"
        )
    else:
        detail = f"{name} 声明支持 {arch_tag} 但实测运算失败：{compute_error}"

    report.add(
        Check(
            f"{arch_tag} 算力支持",
            ok,
            detail,
            required=False,
            data={"device": name, "capability": arch_tag, "arch_list": arch_list},
        )
    )

    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    report.add(
        Check(
            "显存容量",
            total_gb >= 7.5,
            f"{total_gb:.1f} GB" + ("" if total_gb >= 7.5 else "（M3 的 3B 4-bit 部署建议 8GB+）"),
            required=False,
            data={"total_gb": round(total_gb, 2)},
        )
    )


def check_bitsandbytes(report: Report) -> None:
    """4-bit 加载探路。

    用小模型即可——目的是验证 torch + transformers + bitsandbytes 三者
    在当前算力上能协同工作，与具体模型无关。**不看版本号，直接跑**：
    bitsandbytes 的 kernel 需针对新架构单独编译，版本号不能说明问题。
    """
    try:
        import bitsandbytes  # noqa: F401
        import torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError as exc:
        report.add(Check("bitsandbytes 4-bit 加载", False, f"依赖缺失：{exc}", required=False))
        return

    if not torch.cuda.is_available():
        report.add(Check("bitsandbytes 4-bit 加载", False, "CUDA 不可用，跳过", required=False))
        return

    try:
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct", quantization_config=config, device_map="cuda:0"
        )
        used_gb = torch.cuda.memory_allocated() / 1024**3
        del model
        torch.cuda.empty_cache()
        report.add(
            Check(
                "bitsandbytes 4-bit 加载",
                True,
                f"探路模型加载成功，占用 {used_gb:.2f} GB",
                required=False,
                data={"probe_gb": round(used_gb, 2)},
            )
        )
    except Exception as exc:  # noqa: BLE001
        report.add(
            Check(
                "bitsandbytes 4-bit 加载",
                False,
                f"{str(exc).splitlines()[0]}（M3 前必须解决，否则 grounding 模型改在租用 GPU 上部署）",
                required=False,
            )
        )


def check_uia(report: Report) -> None:
    if sys.platform != "win32":
        report.add(Check("UIA 元素树", False, "非 Windows 平台不可用", required=False))
        return
    try:
        from perception.uia_tree import UIATree

        tree = UIATree(max_elements=50, time_budget_ms=2000)
        if not tree.is_available():
            report.add(Check("UIA 元素树", False, "uiautomation 未安装", required=False))
            return
        elements = tree.capture_foreground()
        report.add(
            Check(
                "UIA 元素树",
                len(elements) > 0,
                f"前台窗口抓到 {len(elements)} 个控件，{tree.stats.as_dict()}",
                required=False,
                data=tree.stats.as_dict(),
            )
        )
    except Exception as exc:  # noqa: BLE001
        report.add(Check("UIA 元素树", False, str(exc), required=False))


# ====================================================================== #


def render(report: Report) -> None:
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="开发环境自检", show_lines=False)
        table.add_column("", width=3)
        table.add_column("检查项", style="bold")
        table.add_column("级别", width=6)
        table.add_column("结果")

        for check in report.checks:
            if check.passed:
                mark, style = "[green]✓[/green]", ""
            elif check.required:
                mark, style = "[red]✗[/red]", "red"
            else:
                mark, style = "[yellow]![/yellow]", "yellow"
            level = "必过" if check.required else "警告"
            table.add_row(mark, check.name, level, f"[{style}]{check.detail}[/{style}]" if style else check.detail)

        console.print(table)
        if report.required_failures:
            console.print(
                f"\n[bold red]必过项失败 {len(report.required_failures)} 项，M1 验收不通过[/bold red]"
            )
        else:
            console.print("\n[bold green]必过项全部通过[/bold green]")
        if report.warnings:
            console.print(
                f"[yellow]警告 {len(report.warnings)} 项——不阻塞当前阶段，"
                f"但 M3 开工前必须全部转绿[/yellow]"
            )
    except ImportError:
        for check in report.checks:
            mark = "OK  " if check.passed else ("FAIL" if check.required else "WARN")
            print(f"[{mark}] {check.name}: {check.detail}")
        print()
        print("必过项失败" if report.required_failures else "必过项全部通过")


def main() -> int:
    parser = argparse.ArgumentParser(description="开发环境自检")
    parser.add_argument("--json", action="store_true", help="输出 JSON，供 CI 消费")
    parser.add_argument("--skip-ocr", action="store_true", help="跳过 OCR 检查（首次会下载模型）")
    parser.add_argument("--skip-gpu", action="store_true", help="跳过 GPU 与 bitsandbytes 检查")
    args = parser.parse_args()

    report = Report()
    check_python(report)
    check_dependencies(report)
    check_dpi(report)
    check_screenshot_speed(report)
    check_uia(report)
    check_ocr_cpu(report, skip=args.skip_ocr)
    if not args.skip_gpu:
        check_gpu(report)
        check_bitsandbytes(report)

    if args.json:
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    else:
        render(report)

    return 1 if report.required_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
