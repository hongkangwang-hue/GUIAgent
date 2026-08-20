"""OpenAI 兼容端点的多模态后端 —— 三家平台共用这一份实现。

## 为什么走 ChatOpenAI + base_url

M2 文档划死了这条：``langchain-community`` 里各家国产模型的封装
（``ChatTongyi`` / ``ChatZhipuAI``）对图像内容块的支持成熟度不一——构造
方式、base64 还是 URL、多图顺序各有差异。而截图传递是主链路，**每步都要
走**，不能赌。

``ChatOpenAI(base_url=...)`` 走的是 LangChain 里被验证最充分的那条代码
路径，三家平台都提供兼容端点，因此一份实现覆盖全部。

## 送给模型的图必须和坐标系同尺寸

这是最容易错、错了最难查的一处。

模型是在**它看到的那张图**的像素坐标系里作答的。如果送 2560×1600 的原图
却按 1024×768 的 ``planner`` 坐标系去反算真实坐标，每一次点击都会偏，而
且偏得很有规律——看起来像"模型定位不准"，实际是坐标系错配。

因此这里在发送前把截图缩到 ``image_size``，并且**要求它与 CoordinateScaler
里注册的坐标系尺寸一致**。不一致时直接告警，别等到点歪了再查。

## 图片编码：JPEG 而不是 PNG

截图的 PNG 动辄几 MB。三个理由选 JPEG：

1. NVIDIA NIM 对内联图片有硬上限（见 `providers.NVIDIA_INLINE_LIMIT`），
   超了整条请求被拒
2. 上传字节数直接进 token 账单
3. GUI 截图是大片纯色 + 细小文字，JPEG 在 q=85 上对文字的损伤肉眼难辨

平台有字节上限时，先降质量再降分辨率——**降分辨率会同步破坏坐标系**，
所以它是最后手段，且降了必须告警。
"""

from __future__ import annotations

import base64
import logging
import time
from typing import TYPE_CHECKING, Any

from llm.base import (
    ActionIntent,
    HistoryStep,
    LLMBackend,
    LLMBackendError,
    TokenUsage,
)
from llm.parsing import OutputParseError, parse_action_payload
from llm.providers import ProviderConfig, resolve

if TYPE_CHECKING:  # pragma: no cover - 仅供类型检查
    from perception.capture import Screenshot

logger = logging.getLogger(__name__)

#: 送模型的默认图片尺寸。与 M1 定的 planner 坐标系一致
DEFAULT_IMAGE_SIZE = (1024, 768)

#: JPEG 初始质量。85 是 GUI 截图上文字可读性与体积的经验平衡点
DEFAULT_JPEG_QUALITY = 85

#: 降质量时的下限。再低文字就开始糊了，宁可降分辨率并告警
MIN_JPEG_QUALITY = 45


# ---------------------------------------------------------------------- #
# 错误归类
# ---------------------------------------------------------------------- #

#: 异常文本里出现这些词，判为可重试。
#:
#: 按**文本**而不是异常类型来判，是因为三家平台经由 openai SDK 抛出的
#: 异常类型不完全一致，而 HTTP 语义是一致的。宁可判宽一点：多重试一次
#: 的代价是几秒钟，判漏的代价是一条本可救回的轨迹直接失败。
RETRYABLE_HINTS = (
    "timeout",
    "timed out",
    "rate limit",
    "ratelimit",
    "429",
    "too many requests",
    "502",
    "503",
    "504",
    "bad gateway",
    "service unavailable",
    "gateway timeout",
    "connection",
    "temporarily",
)

#: 这些一看就别重试——重试多少次都一样，只会多烧时间。
#:
#: **顺序有意义**，dict 按插入序遍历，先命中的先返回。额度类排在鉴权
#: 之前是因为两者的 HTTP 状态码会撞：百炼的免费额度耗尽返回的是
#: ``403 AllocationQuota.FreeTierOnly``，若先判 auth，"403" 一命中就
#: 会把一个充值就能解决的问题报成"API key 无效"，让人去查错方向。
#: 两者都不可重试，所以不影响控制流——但会影响 M3 的失败原因统计。
FATAL_HINTS = {
    "quota": (
        "余额不足",
        "欠费",
        "insufficient",
        "balance",
        "quota exceeded",
        "quota exhausted",
        "allocationquota",
        "freetieronly",
        "arrearage",
        "billing",
    ),
    "auth": ("unauthorized", "invalid api key", "authentication", "401", "403", "forbidden"),
    "bad_request": ("invalid_request", "400", "model not found", "does not exist"),
}


def classify_error(exc: Exception) -> tuple[str, bool]:
    """(错误种类, 是否可重试)。

    先判致命再判可重试：限流响应里也可能带 "connection" 字样，顺序反了
    会把余额不足当成网络抖动，一直重试到把重试次数耗光。
    """
    text = f"{type(exc).__name__} {exc}".lower()

    for kind, hints in FATAL_HINTS.items():
        if any(hint in text for hint in hints):
            return kind, False

    if any(hint in text for hint in RETRYABLE_HINTS):
        return "transient", True

    return "unknown", False


# ---------------------------------------------------------------------- #
# 图片编码
# ---------------------------------------------------------------------- #


def encode_screenshot(
    screenshot: Screenshot,
    size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    max_bytes: int = 0,
    quality: int = DEFAULT_JPEG_QUALITY,
) -> tuple[str, dict]:
    """截图 → data URL。返回 (data_url, 元信息)。

    元信息里记了实际尺寸、质量、字节数，**要进轨迹日志**：图被压到多小、
    有没有被迫降分辨率，是排查"模型看不清"这类问题的第一手证据。
    """
    import cv2

    width, height = size
    image = screenshot.resize_to(width, height)
    used_quality = quality
    downscale = 1.0

    payload = _encode_jpeg(cv2, image, used_quality)

    if max_bytes:
        # 先降质量。降到下限还超，再降分辨率——降分辨率会破坏坐标系，
        # 是最后手段
        while len(payload) > max_bytes and used_quality > MIN_JPEG_QUALITY:
            used_quality = max(MIN_JPEG_QUALITY, used_quality - 10)
            payload = _encode_jpeg(cv2, image, used_quality)

        while len(payload) > max_bytes and downscale > 0.35:
            downscale *= 0.8
            scaled = cv2.resize(
                image,
                (max(int(width * downscale), 1), max(int(height * downscale), 1)),
                interpolation=cv2.INTER_AREA,
            )
            payload = _encode_jpeg(cv2, scaled, used_quality)

        if downscale < 1.0:
            logger.warning(
                "为满足 %d 字节上限，截图被降到 %.0f%% 分辨率。"
                "模型看到的图与 planner 坐标系不再等比，定位精度会下降",
                max_bytes,
                downscale * 100,
            )
        if len(payload) > max_bytes:
            logger.error(
                "截图压到 %d 字节仍超过上限 %d，请求很可能被平台拒绝",
                len(payload),
                max_bytes,
            )

    meta = {
        "sent_width": width,
        "sent_height": height,
        "jpeg_quality": used_quality,
        "downscale": round(downscale, 3),
        "bytes": len(payload),
    }
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}", meta


def _encode_jpeg(cv2, image, quality: int) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise LLMBackendError("截图 JPEG 编码失败", kind="encode_error")
    return buffer.tobytes()


# ---------------------------------------------------------------------- #
# 后端
# ---------------------------------------------------------------------- #


class OpenAICompatBackend(LLMBackend):
    """经 LangChain 的 `ChatOpenAI` 调用任意 OpenAI 兼容的多模态端点。"""

    name = "openai_compat"

    def __init__(
        self,
        config: ProviderConfig | None = None,
        system_prompt: str = "",
        few_shot: list[dict] | None = None,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 90.0,
        history_images: int = 1,
    ) -> None:
        self.config = config or resolve()
        super().__init__(model=self.config.model, price=self.config.price)
        self.name = self.config.name

        self.system_prompt = system_prompt
        self.few_shot = few_shot or []
        self.image_size = image_size
        #: 温度取 0：这是动作决策，不是创作。同一个界面同一个目标，
        #: 应当尽量给出同一个动作——否则 M3 的横评每跑一次结论都不同
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        #: 历史里带图的步数。旧帧的边际价值衰减很快，而每张图都按 token
        #: 计费。默认只给最近 1 张，更早的历史用文字摘要代替
        self.history_images = history_images

        self._client = None
        #: 最近一次的图片编码信息，Loop 可取来进轨迹
        self.last_image_meta: dict = {}

    # ------------------------------------------------------------------ #

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - 依赖缺失
            raise LLMBackendError(
                "缺少 langchain-openai，请先 pip install langchain-openai",
                kind="missing_dependency",
            ) from exc

        self._client = ChatOpenAI(
            model=self.config.model,
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            max_retries=0,  # 重试由 AgentLoop 统一管，两层重试会把成本翻倍
        )
        logger.info(
            "已连接 %s：model=%s base_url=%s",
            self.config.provider.label,
            self.config.model,
            self.config.base_url,
        )
        return self._client

    # ------------------------------------------------------------------ #

    def predict_action(
        self,
        instruction: str,
        screenshot: Screenshot,
        history: list[HistoryStep] | None = None,
    ) -> ActionIntent:
        client = self._ensure_client()
        messages = self.build_messages(instruction, screenshot, history or [])

        start = time.perf_counter()
        try:
            response = client.invoke(messages)
        except Exception as exc:  # noqa: BLE001 - SDK 可能抛任何东西
            kind, retryable = classify_error(exc)
            raise LLMBackendError(
                f"{self.config.provider.label} 调用失败（{kind}）：{exc}",
                retryable=retryable,
                kind=kind,
            ) from exc
        latency_ms = (time.perf_counter() - start) * 1000.0

        text = _response_text(response)
        usage = _extract_usage(response)
        cost = self.record_usage(usage)

        try:
            payload = parse_action_payload(text)
        except OutputParseError as exc:
            # **解析失败要带着原文抛。** 没有原文就没法判断是模型的问题
            # 还是解析层的问题，而这正是 M3 比较各模型格式稳定性的依据
            raise LLMBackendError(
                f"模型输出无法解析：{exc}",
                retryable=True,
                kind="parse_error",
            ) from exc

        return ActionIntent(
            **payload,
            raw_text=text,
            usage=usage,
            cost_cny=cost,
            request_id=_request_id(response),
            latency_ms=latency_ms,
        )

    # ------------------------------------------------------------------ #
    # 消息构造
    # ------------------------------------------------------------------ #

    def build_messages(
        self,
        instruction: str,
        screenshot: Screenshot,
        history: list[HistoryStep],
    ) -> list:
        """拼出这一轮要发的消息列表。

        顺序：系统提示 → few-shot 示例 → 历史 → 当前截图与目标。

        few-shot 放在系统提示之后、历史之前，是因为它属于"规则说明"的一
        部分；混进历史里模型会以为那是本次任务真发生过的步骤。
        """
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        messages: list = []
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))

        for example in self.few_shot:
            messages.append(HumanMessage(content=str(example.get("input", ""))))
            messages.append(AIMessage(content=str(example.get("output", ""))))

        messages.extend(self._history_messages(history))

        image_url, self.last_image_meta = encode_screenshot(
            screenshot,
            size=self.image_size,
            max_bytes=self.config.max_image_bytes,
        )
        messages.append(
            HumanMessage(
                content=[
                    {"type": "text", "text": self._user_prompt(instruction)},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]
            )
        )
        return messages

    def _history_messages(self, history: list[HistoryStep]) -> list:
        """历史步骤。只有最近若干步带图，更早的用文字摘要。

        每张图都要按 token 计费，而三步之前的界面对当前决策几乎没有价值。
        文字摘要保留了"做过什么、成没成"这条关键信息，几乎不花钱。
        """
        from langchain_core.messages import AIMessage, HumanMessage

        if not history:
            return []

        messages: list = []
        with_image_from = max(0, len(history) - self.history_images)

        for index, step in enumerate(history):
            messages.append(AIMessage(content=step.summary()))

            if index < with_image_from or step.screenshot is None:
                continue
            image_url, _ = encode_screenshot(
                step.screenshot,
                size=self.image_size,
                max_bytes=self.config.max_image_bytes,
            )
            messages.append(
                HumanMessage(
                    content=[
                        {"type": "text", "text": "执行后的界面："},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ]
                )
            )
        return messages

    def _user_prompt(self, instruction: str) -> str:
        return (
            f"当前子任务目标：{instruction}\n\n"
            f"这是当前屏幕（{self.image_size[0]}×{self.image_size[1]}）。"
            f"请判断下一步动作，坐标必须落在这张图的范围内。"
            f'若该子任务已完成，返回 {{"done": true}}。'
        )

    # ------------------------------------------------------------------ #

    def close(self) -> None:
        self._client = None

    def __repr__(self) -> str:
        return (
            f"<OpenAICompatBackend {self.config.provider.label} "
            f"model={self.config.model!r} image={self.image_size}>"
        )


# ---------------------------------------------------------------------- #
# 响应拆解
# ---------------------------------------------------------------------- #


def _response_text(response: Any) -> str:
    """把响应的文本部分抠出来。

    ``content`` 可能是字符串，也可能是内容块列表（带 reasoning 的模型会
    这样）。两种都要认。
    """
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(parts)
    return str(content)


def _extract_usage(response: Any) -> TokenUsage:
    """从响应里读 token 用量。

    LangChain 把它放在 ``usage_metadata``，但各平台经兼容层透传上来的
    字段名不完全一致，因此再从 ``response_metadata.token_usage`` 兜一道。
    读不到就记 0——**不估算**，估出来的 token 数会污染成本实测数据。
    """
    usage = getattr(response, "usage_metadata", None) or {}
    if usage:
        details = usage.get("input_token_details") or {}
        return TokenUsage(
            prompt_tokens=int(usage.get("input_tokens", 0) or 0),
            completion_tokens=int(usage.get("output_tokens", 0) or 0),
            cached_tokens=int(details.get("cache_read", 0) or 0),
        )

    metadata = getattr(response, "response_metadata", None) or {}
    raw = metadata.get("token_usage") or metadata.get("usage") or {}
    if not raw:
        logger.debug("响应里没有 token 用量信息，本次记 0")
        return TokenUsage()

    details = raw.get("prompt_tokens_details") or {}
    return TokenUsage(
        prompt_tokens=int(raw.get("prompt_tokens", 0) or 0),
        completion_tokens=int(raw.get("completion_tokens", 0) or 0),
        cached_tokens=int(details.get("cached_tokens", 0) or 0),
    )


def _request_id(response: Any) -> str:
    metadata = getattr(response, "response_metadata", None) or {}
    for key in ("id", "request_id", "x-request-id"):
        if metadata.get(key):
            return str(metadata[key])
    return str(getattr(response, "id", "") or "")
