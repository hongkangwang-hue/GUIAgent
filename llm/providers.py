"""模型平台注册表 —— 三家 OpenAI 兼容端点的差异都收在这里。

## 为什么不是 QwenVLAPIBackend

M2 文档原计划写一个 `QwenVLAPIBackend`，M3 再补 GLM-4V 与 Llama 的实现。
但实际可用的三家平台**都提供 OpenAI 兼容端点**：

| 平台 | 模型 | 端点 |
|---|---|---|
| 阿里云百炼 | qwen3-vl-8b-instruct | dashscope.aliyuncs.com/compatible-mode/v1 |
| 智谱开放平台 | glm-4.6v-flash | open.bigmodel.cn/api/paas/v4 |
| NVIDIA NIM | meta/llama-3.2-11b-vision-instruct | integrate.api.nvidia.com/v1 |

既然协议是同一个，写三个类就是把同一份逻辑抄三遍——而 M3 的横评恰恰
要求三者跑在**完全相同的代码路径**上，否则跑出来的差异分不清是模型带来
的还是实现带来的。所以只有一个 `OpenAICompatBackend`，平台差异全部降级
成这张表里的数据。

M2 验收标准第 8 条"新增一个模型后端不需要改动 Agent 层与 Loop 层"，在
这个结构下变成了更强的一条：**新增平台连后端类都不用改，加一条记录即可。**

## 单价为什么默认是 None

项目规矩：不猜单价。假成本比"未知"更有害——M0 的成本估算要靠 M2 的实测
数据校准，掺了猜测就白测了。

代价是成本熔断在没配单价时不生效（`AgentLoop._over_budget` 会跳过）。
这是有意的取舍，且已在 `CostInfo.priced` 上标明。要启用熔断，就去平台
的计费页面查到真实单价填进 .env，那才是能进报告的数字。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from llm.base import PriceSheet

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Provider:
    """一个模型平台的接入参数与已知限制。"""

    key: str
    label: str
    #: 环境变量名
    api_key_env: str
    base_url_env: str
    model_env: str
    default_base_url: str
    default_model: str

    #: 内联图片的字节上限。超过要降质量或降分辨率。
    #: 0 表示没有已知限制
    max_image_bytes: int = 0

    #: 该平台已知的坑，进日志与报告
    notes: str = ""


#: 单个 base64 内联图片的保守上限。
#:
#: NVIDIA NIM 的文档写明：内联图片超过 180KB 就必须改走 NVCF 资产上传
#: 接口，那是另一套协议，不在 OpenAI 兼容路径上。桌面截图很容易超——
#: 2560×1600 的 PNG 动辄几 MB。因此这里按 JPEG 编码并自适应降质量。
#:
#: 取 150KB 而不是 180KB：base64 编码会让体积涨约 33%，留出余量。
NVIDIA_INLINE_LIMIT = 150 * 1024


PROVIDERS: dict[str, Provider] = {
    "dashscope": Provider(
        key="dashscope",
        label="阿里云百炼",
        api_key_env="DASHSCOPE_API_KEY",
        base_url_env="DASHSCOPE_BASE_URL",
        model_env="PLANNER_MODEL",
        default_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        default_model="qwen3-vl-8b-instruct",
        notes="OpenAI 兼容模式。Qwen-VL 系列原生支持 grounding，是模式 A 的主力",
    ),
    "zhipu": Provider(
        key="zhipu",
        label="智谱开放平台",
        api_key_env="ZHIPU_API_KEY",
        base_url_env="ZHIPU_BASE_URL",
        model_env="ZHIPU_MODEL",
        default_base_url="https://open.bigmodel.cn/api/paas/v4",
        default_model="glm-4.6v-flash",
        notes="flash 档为免费或低价模型，适合做横评的成本对照组",
    ),
    "nvidia": Provider(
        key="nvidia",
        label="NVIDIA NIM",
        api_key_env="NVIDIA_API_KEY",
        base_url_env="NVIDIA_BASE_URL",
        model_env="NVIDIA_MODEL",
        default_base_url="https://integrate.api.nvidia.com/v1",
        default_model="meta/llama-3.2-11b-vision-instruct",
        max_image_bytes=NVIDIA_INLINE_LIMIT,
        notes=(
            "内联图片有硬上限，超过要走 NVCF 资产上传（另一套协议）。"
            "桌面截图必须压到限制内，否则整条请求被拒"
        ),
    ),
}

#: 没指定平台时用哪个
DEFAULT_PROVIDER = "dashscope"


class ProviderNotConfigured(RuntimeError):
    """平台缺少必要配置（通常是 API key）。"""


@dataclass
class ProviderConfig:
    """解析完环境变量之后的实际接入参数。"""

    provider: Provider
    api_key: str
    base_url: str
    model: str
    price: PriceSheet | None = None

    @property
    def name(self) -> str:
        return self.provider.key

    @property
    def max_image_bytes(self) -> int:
        return self.provider.max_image_bytes

    def masked(self) -> dict:
        """可以安全打进日志与 `config --show` 的形式。

        **key 只留头尾**。轨迹日志、控制台输出、贴给别人看的报错，都可能
        把这个 dict 带出去；一次疏忽就是一把废掉的凭据。
        """
        return {
            "provider": self.provider.key,
            "label": self.provider.label,
            "model": self.model,
            "base_url": self.base_url,
            "api_key": _mask(self.api_key),
            "priced": self.price is not None,
        }


def _mask(secret: str) -> str:
    if not secret:
        return "（未配置）"
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:4]}…{secret[-4:]}"


def _price_from_env(provider: Provider) -> PriceSheet | None:
    """从 .env 读单价。没配就返回 None —— 不猜。

    变量名形如 ``DASHSCOPE_PRICE_IN_PER_1K`` / ``DASHSCOPE_PRICE_OUT_PER_1K``，
    单位元/千 token，去平台计费页面抄。
    """
    prefix = provider.api_key_env.rsplit("_API_KEY", 1)[0]
    raw_in = os.getenv(f"{prefix}_PRICE_IN_PER_1K")
    raw_out = os.getenv(f"{prefix}_PRICE_OUT_PER_1K")
    if raw_in is None and raw_out is None:
        return None
    try:
        return PriceSheet(
            model=provider.default_model,
            input_per_1k=float(raw_in or 0.0),
            output_per_1k=float(raw_out or 0.0),
            cached_input_per_1k=_optional_float(f"{prefix}_PRICE_CACHED_IN_PER_1K"),
        )
    except ValueError:
        logger.warning("%s 的单价配置不是数字，按未配置处理", provider.label)
        return None


def _optional_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def load_dotenv_if_present(path: str = ".env") -> bool:
    """把 .env 读进环境变量。

    自己解析而不是拉 python-dotenv：需求只有"KEY=VALUE 一行一条"，
    为此多一个依赖不划算。**已存在的环境变量不覆盖**——命令行里显式
    设的值应当优先于文件。
    """
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    return True


def resolve(
    name: str | None = None,
    model: str | None = None,
    require_key: bool = True,
) -> ProviderConfig:
    """按平台名解析出可用的接入参数。

    ``model`` 显式传入时覆盖环境变量——横评要在同一平台上换模型跑。
    """
    key = (name or os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
    if key not in PROVIDERS:
        raise ProviderNotConfigured(f"未知平台 {key!r}。已注册：{', '.join(sorted(PROVIDERS))}")
    provider = PROVIDERS[key]

    api_key = os.getenv(provider.api_key_env, "").strip()
    if require_key and not api_key:
        raise ProviderNotConfigured(
            f"{provider.label} 缺少 API key：请在 .env 里设置 {provider.api_key_env}。"
            f"（.env 已在 .gitignore 中，不会被提交）"
        )

    return ProviderConfig(
        provider=provider,
        api_key=api_key,
        base_url=os.getenv(provider.base_url_env, "").strip() or provider.default_base_url,
        model=(model or os.getenv(provider.model_env, "").strip() or provider.default_model),
        price=_price_from_env(provider),
    )


def available_providers() -> list[tuple[str, bool]]:
    """(平台名, key 是否已配置)。`config --show` 与选型脚本用。"""
    return [
        (key, bool(os.getenv(provider.api_key_env, "").strip()))
        for key, provider in sorted(PROVIDERS.items())
    ]
