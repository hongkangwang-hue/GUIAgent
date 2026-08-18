"""OCR 引擎抽象与两个实现。

抽象出 `OCREngine` 的目的不是"以后可能换引擎"这种空泛的扩展性，而是
M1 交付物里明确要求的**双引擎对照实验**（PaddleOCR vs EasyOCR）：两个
引擎必须跑在完全相同的预处理与后处理下，对比才有意义。共享的部分放在
基类里，各自只实现 `_raw_recognize`。

两个引擎都**延迟导入**：PaddleOCR 与 EasyOCR 首次导入都要几秒并加载
模型权重，不该在 `import perception` 时就付出这个代价。
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from perception.preprocess import DEFAULT, PreprocessConfig, preprocess
from perception.types import BBox

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """一条 OCR 结果，坐标已还原到**原图**坐标系（超分已抵消）。"""

    text: str
    bbox: BBox
    confidence: float
    #: 原始多边形顶点，倾斜文本时比外接矩形更精确，留作调试
    polygon: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class OCRBenchmark:
    """单次识别的性能与产出摘要，双引擎对照实验的原始记录。"""

    engine: str
    elapsed_ms: float
    num_results: int
    mean_confidence: float
    preprocess_steps: list[str]


class OCREngine(ABC):
    """OCR 引擎基类。

    子类只需实现 `_raw_recognize`，预处理、坐标还原、低置信过滤、耗时
    统计都由基类统一处理——这样两个引擎的对比才是公平的。
    """

    name: str = "base"

    def __init__(
        self,
        config: PreprocessConfig = DEFAULT,
        min_confidence: float = 0.5,
    ) -> None:
        self.config = config
        self.min_confidence = min_confidence
        self._last_benchmark: OCRBenchmark | None = None

    # ------------------------------------------------------------------ #

    @abstractmethod
    def _raw_recognize(self, image: np.ndarray) -> list[OCRResult]:
        """在**已预处理**的图上识别，返回该图坐标系下的结果。"""

    @abstractmethod
    def is_available(self) -> bool:
        """依赖是否已安装且模型可加载。"""

    # ------------------------------------------------------------------ #

    def recognize(self, image: np.ndarray) -> list[OCRResult]:
        """识别图中文字，返回**原图坐标系**下的结果。

        坐标还原由基类负责：预处理里的 2× 超分会让所有坐标翻倍，忘记
        除回去是这类流水线最常见的坐标 bug，因此绝不留给子类处理。
        """
        start = time.perf_counter()
        processed = preprocess(image, self.config)
        raw = self._raw_recognize(processed)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        scale = self.config.upscale
        results = [
            self._rescale(item, scale)
            for item in raw
            if item.confidence >= self.min_confidence and item.text.strip()
        ]

        confidences = [r.confidence for r in results]
        self._last_benchmark = OCRBenchmark(
            engine=self.name,
            elapsed_ms=round(elapsed_ms, 2),
            num_results=len(results),
            mean_confidence=round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
            preprocess_steps=self.config.enabled_steps(),
        )
        logger.debug("%s 识别 %d 条，耗时 %.1fms", self.name, len(results), elapsed_ms)
        return results

    @property
    def last_benchmark(self) -> OCRBenchmark | None:
        return self._last_benchmark

    @staticmethod
    def _rescale(result: OCRResult, scale: float) -> OCRResult:
        if scale == 1.0:
            return result
        box = result.bbox
        return OCRResult(
            text=result.text,
            bbox=BBox(
                int(box.left / scale),
                int(box.top / scale),
                max(int(box.right / scale), int(box.left / scale) + 1),
                max(int(box.bottom / scale), int(box.top / scale) + 1),
            ),
            confidence=result.confidence,
            polygon=[(int(x / scale), int(y / scale)) for x, y in result.polygon],
        )

    @staticmethod
    def _bbox_from_polygon(points) -> tuple[BBox, list[tuple[int, int]]]:
        """多边形顶点 → 外接矩形。两个引擎给的都是四点多边形。"""
        xs = [int(p[0]) for p in points]
        ys = [int(p[1]) for p in points]
        polygon = [(int(p[0]), int(p[1])) for p in points]
        return BBox(min(xs), min(ys), max(xs) + 1, max(ys) + 1), polygon


# ====================================================================== #


class PaddleOCREngine(OCREngine):
    """主 OCR 引擎。中文识别效果与速度综合最好。"""

    name = "paddleocr"

    def __init__(
        self,
        config: PreprocessConfig = DEFAULT,
        min_confidence: float = 0.5,
        lang: str = "ch",
        use_gpu: bool = False,
    ) -> None:
        super().__init__(config, min_confidence)
        self.lang = lang
        #: 默认 CPU。GPU 相关检查在 M1 是警告项，OCR 不该被 Blackwell
        #: 工具链问题阻塞（见 M1 前置状态确认）
        self.use_gpu = use_gpu
        self._ocr = None

    def _ensure_loaded(self):
        if self._ocr is not None:
            return self._ocr
        from paddleocr import PaddleOCR

        # PaddleOCR 2.x 与 3.x 的构造参数不同，逐个试而不是猜版本
        for kwargs in (
            {"use_angle_cls": True, "lang": self.lang, "use_gpu": self.use_gpu, "show_log": False},
            {"use_angle_cls": True, "lang": self.lang, "use_gpu": self.use_gpu},
            {"lang": self.lang},
        ):
            try:
                self._ocr = PaddleOCR(**kwargs)
                logger.info("PaddleOCR 已加载：%s", kwargs)
                return self._ocr
            except (TypeError, ValueError) as exc:
                last = exc
        raise RuntimeError(f"PaddleOCR 构造失败，请检查版本：{last}")

    def is_available(self) -> bool:
        try:
            self._ensure_loaded()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("PaddleOCR 不可用：%s", exc)
            return False

    def _raw_recognize(self, image: np.ndarray) -> list[OCRResult]:
        ocr = self._ensure_loaded()

        # 二值化后是单通道，PaddleOCR 需要三通道
        if image.ndim == 2:
            import cv2

            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        try:
            raw = ocr.ocr(image, cls=True)
        except TypeError:
            raw = ocr.ocr(image)  # 3.x 去掉了 cls 参数

        return self._parse(raw)

    @staticmethod
    def _parse(raw) -> list[OCRResult]:
        """解析 PaddleOCR 输出。

        2.x 返回 ``[[ [poly, (text, conf)], ... ]]``（外层按图分组），
        某些版本在无结果时返回 ``[None]``。这里都当成异常形状兜住——
        OCR 库的输出格式在版本间变过好几次，写死一种解析必然踩雷。
        """
        if not raw:
            return []

        lines = raw[0] if isinstance(raw[0], list) or raw[0] is None else raw
        if not lines:
            return []

        results: list[OCRResult] = []
        for line in lines:
            try:
                polygon, (text, confidence) = line[0], line[1]
                bbox, poly = OCREngine._bbox_from_polygon(polygon)
                results.append(
                    OCRResult(
                        text=str(text), bbox=bbox, confidence=float(confidence), polygon=poly
                    )
                )
            except (TypeError, ValueError, IndexError) as exc:
                logger.debug("跳过无法解析的 PaddleOCR 结果 %r：%s", line, exc)
        return results


# ====================================================================== #


class EasyOCREngine(OCREngine):
    """对照引擎。M1 交付物《OCR 双引擎对照实验小结》的另一半。"""

    name = "easyocr"

    def __init__(
        self,
        config: PreprocessConfig = DEFAULT,
        min_confidence: float = 0.5,
        languages: tuple[str, ...] = ("ch_sim", "en"),
        use_gpu: bool = False,
    ) -> None:
        super().__init__(config, min_confidence)
        self.languages = list(languages)
        self.use_gpu = use_gpu
        self._reader = None

    def _ensure_loaded(self):
        if self._reader is None:
            import easyocr

            self._reader = easyocr.Reader(self.languages, gpu=self.use_gpu, verbose=False)
            logger.info("EasyOCR 已加载：%s", self.languages)
        return self._reader

    def is_available(self) -> bool:
        try:
            self._ensure_loaded()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("EasyOCR 不可用：%s", exc)
            return False

    def _raw_recognize(self, image: np.ndarray) -> list[OCRResult]:
        reader = self._ensure_loaded()
        results: list[OCRResult] = []
        for polygon, text, confidence in reader.readtext(image):
            bbox, poly = OCREngine._bbox_from_polygon(polygon)
            results.append(
                OCRResult(text=str(text), bbox=bbox, confidence=float(confidence), polygon=poly)
            )
        return results


# ====================================================================== #


def compare_engines(
    image: np.ndarray,
    configs: dict | None = None,
) -> dict:
    """双引擎对照实验的执行入口。

    在同一张图、同一套预处理下跑两个引擎，返回可直接进表格的记录。
    M1 交付物《OCR 双引擎对照实验小结》要求记录中文准确率、英文准确率、
    平均耗时——准确率需要人工标注真值，本函数负责产出**耗时与产出量**
    这部分客观数据，准确率由标注脚本另行计算。
    """
    configs = configs or {"default": DEFAULT}
    report: dict = {}

    for config_name, config in configs.items():
        for engine in (PaddleOCREngine(config), EasyOCREngine(config)):
            if not engine.is_available():
                report[f"{engine.name}/{config_name}"] = {"available": False}
                continue
            results = engine.recognize(image)
            benchmark = engine.last_benchmark
            report[f"{engine.name}/{config_name}"] = {
                "available": True,
                "elapsed_ms": benchmark.elapsed_ms,
                "num_results": benchmark.num_results,
                "mean_confidence": benchmark.mean_confidence,
                "preprocess": benchmark.preprocess_steps,
                "texts": [r.text for r in results],
            }
    return report
