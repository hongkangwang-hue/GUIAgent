"""提示词模板库 —— YAML 外置、版本化、可消融。

## 为什么外置成 YAML

M3 任务 3 要做提示词消融：三个版本（零样本基线 / +角色设定+few-shot /
+CoT 显式推理）各跑 45 次，比成功率。**如果提示词写死在代码里，"换一个
版本"就意味着改代码重跑**——那样的对比不成立，因为你没法保证除了提示词
之外别的都没变。

外置之后，消融是 `--prompt-version p2` 一个参数的事。

## 动作说明必须从 ACTION_SPECS 生成，不能手写在 YAML 里

M1 的 `control/actions.py` 开头立了规矩：

> JSON Schema 与参数校验都从这张表生成，保证"给模型看的"和"实际校验的"
> 永远是同一份定义。

如果在 YAML 里手抄一份动作列表，改了某个动作的参数名，YAML 不会跟着变。
模型照着过时的说明输出，然后在 `Action.validate` 那里被拒——而且报错
指向的是模型不听话，不是提示词过期。这种 bug 极难查。

所以 YAML 只放**叙述性内容**（角色设定、原则、few-shot 示例），动作清单
由 `render_action_reference()` 从 `ACTION_SPECS` 现场生成后注入。

## few-shot 为什么是必需项而不是优化项

M2 设计思路里那句话说得很直白：

> 开源模型的规划能力弱于闭源前沿模型，以下两条不是优化项，是能不能跑
> 起来的前提。

其中第二条就是"提示词内置 few-shot 示例"。因此模板格式里 `few_shot` 是
一等公民，不是可选附加。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from control.actions import ACTION_SPECS, CORE_ACTIONS, ActionType

logger = logging.getLogger(__name__)

#: 模板目录。与代码同级而非包内，方便直接编辑与 diff
PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


class PromptError(RuntimeError):
    """模板缺失或格式不对。"""


@dataclass
class FewShotExample:
    """一组"输入 → 正确输出"示例。

    ``input`` 是**截图的文字描述**而非真实图片：附带真实图片会让每次
    请求多花几千 token，而 few-shot 在这里的作用是**示范输出格式与决策
    风格**，文字描述足够。真实的视觉信息由当前这一步的截图提供。
    """

    input: str
    output: str
    note: str = ""


@dataclass
class PromptTemplate:
    """一个版本化的提示词模板。"""

    name: str
    version: str = "v1"
    description: str = ""
    system: str = ""
    user_template: str = ""
    few_shot: list[FewShotExample] = field(default_factory=list)
    #: 该版本启用了哪些特性，消融报告里按这个分组
    features: list[str] = field(default_factory=list)
    source_path: str = ""

    def render_system(self, **kwargs: Any) -> str:
        """渲染系统提示。

        ``{action_reference}`` 是保留占位符，自动注入由 `ACTION_SPECS`
        生成的动作清单——见模块文档"不能手写"那一节。
        """
        values = {"action_reference": render_action_reference(), **kwargs}
        return _safe_format(self.system, values, where=f"{self.name}.system")

    def render_user(self, **kwargs: Any) -> str:
        return _safe_format(self.user_template, kwargs, where=f"{self.name}.user_template")

    def few_shot_pairs(self) -> list[dict]:
        """转成 `OpenAICompatBackend` 认的形状。"""
        return [{"input": e.input, "output": e.output} for e in self.few_shot]

    def as_dict(self) -> dict:
        """进轨迹日志。M3 消融要能追溯每条轨迹用的是哪个提示词版本。"""
        return {
            "name": self.name,
            "version": self.version,
            "features": list(self.features),
            "few_shot_count": len(self.few_shot),
        }


def _safe_format(template: str, values: dict, where: str) -> str:
    """按 ``{name}`` 占位符替换。

    不用 `str.format`：提示词里大量出现 JSON 示例（``{"action": ...}``），
    `format` 会把那些花括号当占位符，抛 KeyError 或吃掉内容。**而 JSON
    示例恰恰是这个项目提示词里最重要的部分**，不能要求写模板的人把每个
    花括号都转义成 ``{{``——那样模板会变得没法读。

    因此改成只替换**明确声明过的键**，其余花括号原样保留。
    """
    result = template
    for key, value in values.items():
        result = result.replace("{" + key + "}", str(value))

    leftover = [k for k in values if "{" + k + "}" in result]
    if leftover:  # pragma: no cover - 替换后不该还有残留
        logger.warning("%s 中的占位符 %s 未被替换", where, leftover)
    return result


# ---------------------------------------------------------------------- #
# 动作清单
# ---------------------------------------------------------------------- #


def render_action_reference(only_core: bool = True) -> str:
    """从 `ACTION_SPECS` 生成给模型看的动作说明。

    与 `control.actions.to_tool_schemas` 同源，因此动作参数一改，提示词
    自动跟着变——不会出现"提示词说有这个参数、校验说没有"的错位。
    """
    types = sorted(CORE_ACTIONS, key=lambda t: t.value) if only_core else list(ActionType)

    lines: list[str] = []
    for action_type in types:
        description, specs = ACTION_SPECS[action_type]
        if not specs:
            lines.append(f"- `{action_type.value}` —— {description}（无参数）")
            continue

        parts = []
        for spec in specs:
            marker = "" if spec.required else "，可选"
            enum_hint = f"，取值 {'/'.join(spec.enum)}" if spec.enum else ""
            parts.append(f"`{spec.name}`({spec.json_type}{marker}{enum_hint})")
        lines.append(f"- `{action_type.value}` —— {description}。参数：{', '.join(parts)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------- #
# 加载
# ---------------------------------------------------------------------- #


def load_template(name: str, directory: Path | str | None = None) -> PromptTemplate:
    """按名字加载模板。``name`` 形如 ``executor_v1``，对应 ``executor_v1.yaml``。"""
    directory = Path(directory) if directory else PROMPT_DIR
    path = directory / f"{name}.yaml"
    if not path.exists():
        available = ", ".join(sorted(p.stem for p in directory.glob("*.yaml"))) or "（无）"
        raise PromptError(f"找不到提示词模板 {path}。可用：{available}")

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - 依赖缺失
        raise PromptError("缺少 pyyaml，请先 pip install pyyaml") from exc

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise PromptError(f"{path} 的顶层必须是映射")

    examples = []
    for index, raw in enumerate(data.get("few_shot") or []):
        if not isinstance(raw, dict) or "input" not in raw or "output" not in raw:
            raise PromptError(f"{path} 的 few_shot[{index}] 必须同时含 input 与 output")
        examples.append(
            FewShotExample(
                input=str(raw["input"]).strip(),
                output=str(raw["output"]).strip(),
                note=str(raw.get("note", "")).strip(),
            )
        )

    return PromptTemplate(
        name=str(data.get("name", name)),
        version=str(data.get("version", "v1")),
        description=str(data.get("description", "")),
        system=str(data.get("system", "")).strip(),
        user_template=str(data.get("user_template", "")).strip(),
        few_shot=examples,
        features=[str(f) for f in (data.get("features") or [])],
        source_path=str(path),
    )


def list_templates(directory: Path | str | None = None) -> list[str]:
    """列出可用模板名。CLI 的 `config --show` 与消融脚本用。"""
    directory = Path(directory) if directory else PROMPT_DIR
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob("*.yaml"))
