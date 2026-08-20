"""Planner —— 把一句指令拆成一串原子级子任务。

## 为什么必须拆得很细

M2 设计思路里的原话：

> 不要给模型"完成整个任务"这种大目标——它扛不住。Planner 把任务拆成
> 一串小到只需一两步就能完成的子目标，Executor 一次只带**一个**子目标
> 进 Agent Loop。**这一条原计划在 M4，现在必须在 M2 就做。**

提前的原因是开源规划模型的能力边界。子任务越大，模型要在一次决策里
考虑的东西越多，越容易跑偏；而跑偏之后没有反思模块能救（M2 不做
Reflector），只能失败终止。

## 粒度检查是程序化的，不靠肉眼

M2 验收标准 4：「Planner 输出的子任务粒度可验证：每个子任务在 1-2 步内
可完成」。

"可验证"这三个字如果只靠人看，验收时就变成了各说各话。因此这里实现了
`granularity_report()`：用几条可判定的启发式规则给出警告清单——连词、
过长、动词过多。**它不拒绝执行**（启发式必然有误判），但它把"粒度好不好"
变成了一份可以摆出来的记录。
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agent.prompts import PromptTemplate, load_template
from llm.base import LLMBackend, LLMBackendError, TokenUsage
from llm.parsing import OutputParseError, extract_json

if TYPE_CHECKING:  # pragma: no cover - 仅供类型检查
    from perception.capture import Screenshot

logger = logging.getLogger(__name__)

#: 子任务数量上限。M2「明确不做」第 6 条：不做复杂任务（超过 8 步 / 跨应用）
MAX_SUBTASKS = 8

#: 提示"这条塞了不止一件事"的连词。
#:
#: 这是粒度检查里最有效的一条：中文里复合动作几乎总是被这些词连起来的，
#: 而提示词也明确要求过"不要用并且/然后把两件事塞进一个子任务"。
CONJUNCTIONS = ("并且", "然后", "接着", "同时", "之后再", "再把", "以及", "，并", "and then")

#: 单个子任务描述的字数上限。超过通常意味着塞了多件事或写成了叙述
MAX_GOAL_LENGTH = 40

#: GUI 操作动词。一条子任务里出现两个以上，多半是复合动作
ACTION_VERBS = (
    "点击",
    "单击",
    "双击",
    "右键",
    "输入",
    "打开",
    "关闭",
    "保存",
    "选择",
    "拖动",
    "滚动",
    "按下",
    "切换",
    "启动",
    "退出",
    "删除",
)


class PlanError(RuntimeError):
    """拆解失败。"""


@dataclass
class SubTask:
    """一个原子级子目标。"""

    id: int
    goal: str
    #: 完成后屏幕上应该出现的**看得见的**变化。
    #: 要求"看得见"是因为它服务于判断这一步做完没有，而 M2 不做 Reflector，
    #: 判定只能靠人或简单规则看截图——"系统内部状态"那种写法没法用
    expected: str = ""

    def as_dict(self) -> dict:
        return {"id": self.id, "goal": self.goal, "expected": self.expected}


@dataclass
class Plan:
    """一次拆解的完整结果。"""

    instruction: str = ""
    subtasks: list[SubTask] = field(default_factory=list)
    raw_text: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)
    cost_cny: float = 0.0
    latency_ms: float = 0.0
    request_id: str = ""
    #: 使用的模板版本，进轨迹日志。M3 消融要能追溯
    prompt: dict = field(default_factory=dict)
    #: 被截断掉的子任务数（超过 MAX_SUBTASKS 时）
    truncated: int = 0

    def goals(self) -> list[str]:
        return [s.goal for s in self.subtasks]

    def as_dict(self) -> dict:
        return {
            "instruction": self.instruction,
            "subtasks": [s.as_dict() for s in self.subtasks],
            "prompt": dict(self.prompt),
            "truncated": self.truncated,
            "cost_cny": round(self.cost_cny, 6),
        }

    # ------------------------------------------------------------------ #

    def granularity_report(self) -> list[dict]:
        """粒度自检。返回警告清单，空列表表示没发现问题。

        **只警告不拦截。** 这几条都是启发式，必然有误判——"点击并输入"
        确实该拆，但"打开控制面板"里的"打开"是一个动作不是两个。把启发式
        提成硬性拒绝，会在演示时因为一条误判把整个任务卡死。

        它的用途是 M2 验收标准 4 的证据：把这份报告和拆解结果一起摆出来，
        "粒度可验证"就有了具体含义。
        """
        warnings: list[dict] = []
        for task in self.subtasks:
            hits = [word for word in CONJUNCTIONS if word in task.goal]
            if hits:
                warnings.append(
                    {
                        "id": task.id,
                        "rule": "conjunction",
                        "detail": f"含连词 {hits}，可能塞了不止一件事",
                        "goal": task.goal,
                    }
                )

            if len(task.goal) > MAX_GOAL_LENGTH:
                warnings.append(
                    {
                        "id": task.id,
                        "rule": "too_long",
                        "detail": f"描述 {len(task.goal)} 字，超过 {MAX_GOAL_LENGTH} 字",
                        "goal": task.goal,
                    }
                )

            verbs = [v for v in ACTION_VERBS if v in task.goal]
            if len(verbs) > 1:
                warnings.append(
                    {
                        "id": task.id,
                        "rule": "multi_verb",
                        "detail": f"出现多个动作词 {verbs}",
                        "goal": task.goal,
                    }
                )

            if not task.expected:
                warnings.append(
                    {
                        "id": task.id,
                        "rule": "no_expected",
                        "detail": "缺少完成判据，无法判断这一步做完没有",
                        "goal": task.goal,
                    }
                )
        return warnings

    def is_fine_grained(self) -> bool:
        """没有任何粒度警告。CLI 用它决定要不要提醒用户。"""
        return not self.granularity_report()


# ---------------------------------------------------------------------- #


class Planner:
    """调用规划模型做任务拆解。

    与 `AgentLoop` 的分工：Planner 只跑**一次**，在任务开始时把指令拆开；
    之后每个子任务各自进 Loop。拆解用的是同一个 `LLMBackend`，因此成本
    自动并入总账。
    """

    def __init__(
        self,
        backend: LLMBackend,
        template: PromptTemplate | None = None,
        max_subtasks: int = MAX_SUBTASKS,
        with_screenshot: bool = True,
    ) -> None:
        self.backend = backend
        self.template = template or load_template("planner_v1")
        self.max_subtasks = max_subtasks
        #: 拆解时要不要给模型看当前屏幕。
        #:
        #: 看得见更好——"打开记事本"这个指令，在记事本已经开着的情况下
        #: 应该拆出完全不同的步骤。代价是一次图片的 token。
        #: 置 False 则纯文本拆解，M3 消融时可作为一个对照臂。
        self.with_screenshot = with_screenshot

    # ------------------------------------------------------------------ #

    def plan(self, instruction: str, screenshot: Screenshot | None = None) -> Plan:
        """把指令拆成子任务列表。"""
        if not instruction or not instruction.strip():
            raise PlanError("指令为空")

        system = self.template.render_system(width=0, height=0)
        user = self.template.render_user(instruction=instruction.strip())

        start = time.perf_counter()
        try:
            intent = self._ask(system, user, screenshot)
        except LLMBackendError as exc:
            raise PlanError(f"拆解调用失败：{exc}") from exc
        latency_ms = (time.perf_counter() - start) * 1000.0

        subtasks, truncated = self._parse(intent.raw_text)
        plan = Plan(
            instruction=instruction.strip(),
            subtasks=subtasks,
            raw_text=intent.raw_text,
            usage=intent.usage,
            cost_cny=intent.cost_cny,
            latency_ms=intent.latency_ms or latency_ms,
            request_id=intent.request_id,
            prompt=self.template.as_dict(),
            truncated=truncated,
        )

        warnings = plan.granularity_report()
        if warnings:
            logger.warning("拆解结果有 %d 条粒度警告：%s", len(warnings), warnings)
        logger.info("指令 %r 拆成 %d 个子任务", instruction, len(subtasks))
        return plan

    # ------------------------------------------------------------------ #

    def _ask(self, system: str, user: str, screenshot: Screenshot | None):
        """借 `LLMBackend` 发一次请求。

        复用同一个后端而不是另起一路，有两个理由：成本自动并入总账（单
        任务成本是 M2 的交付物）；M3 换后端时 Planner 跟着换，不需要单独
        配置——否则会出现"规划用 A 模型、执行用 B 模型"的隐藏状态，横评
        结论就说不清了。
        """
        saved = (self.backend.system_prompt, self.backend.few_shot, self.backend.user_template)
        try:
            self.backend.system_prompt = system
            self.backend.few_shot = self.template.few_shot_pairs()
            # 拆解阶段的用户消息已经由模板渲染好了，后端不要再包一层
            # "这是当前屏幕…请判断下一步动作"——那是执行阶段的说法，
            # 混进拆解请求里会把模型引向输出单个动作而不是子任务列表
            self.backend.user_template = "{instruction}"
            return self.backend.predict_action(
                user,
                screenshot if self.with_screenshot else None,
                history=None,
            )
        finally:
            (
                self.backend.system_prompt,
                self.backend.few_shot,
                self.backend.user_template,
            ) = saved

    def _parse(self, raw: str) -> tuple[list[SubTask], int]:
        """解析拆解结果。

        容错口径与 `llm.parsing` 一致：模型把列表直接放在顶层、或裹在
        ``subtasks`` / ``plan`` / ``steps`` 里，都认。
        """
        try:
            data = extract_json(raw)
        except OutputParseError as exc:
            raise PlanError(f"拆解结果无法解析：{exc}") from exc

        items = None
        for key in ("subtasks", "plan", "steps", "tasks"):
            if isinstance(data.get(key), list):
                items = data[key]
                break
        if items is None:
            # extract_json 遇到顶层数组时会返回第一个元素，因此这里再兜一道：
            # 模型只给了单个子任务对象的情况
            if "goal" in data or "task" in data:
                items = [data]
            else:
                raise PlanError(f"拆解结果里找不到子任务列表，可用键：{sorted(data)}")

        subtasks: list[SubTask] = []
        for item in items:
            goal = self._goal_of(item)
            if not goal:
                logger.debug("跳过没有目标描述的子任务：%r", item)
                continue
            expected = ""
            if isinstance(item, dict):
                expected = str(
                    item.get("expected") or item.get("expectation") or item.get("result") or ""
                ).strip()
            subtasks.append(SubTask(id=len(subtasks) + 1, goal=goal, expected=expected))

        if not subtasks:
            raise PlanError("拆解结果为空——模型没有给出任何子任务")

        truncated = max(0, len(subtasks) - self.max_subtasks)
        if truncated:
            # M2「明确不做」第 6 条：不做复杂任务（超过 8 步）。截断而不是
            # 报错——前 8 步照样能演示，且截断数会进轨迹，事后看得出来
            logger.warning("拆出 %d 个子任务，超过上限 %d，截断", len(subtasks), self.max_subtasks)
            subtasks = subtasks[: self.max_subtasks]

        return subtasks, truncated

    @staticmethod
    def _goal_of(item) -> str:
        if isinstance(item, str):
            return item.strip()
        if not isinstance(item, dict):
            return ""
        for key in ("goal", "task", "subtask", "description", "step", "action"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return _strip_numbering(value)
        return ""


#: 匹配 "1. " / "1、" / "(2)" / "第3步：" / "第 4 条:" 这类序号前缀。
#: 中间那个可选的量词（步/条/个/项）是实测踩到的——模型很爱写"第2步："，
#: 而只匹配"第2："的模式会漏掉它。
_NUMBER_PREFIX = re.compile(r"^\s*[（(]?\s*第?\s*\d+\s*[步条个项]?\s*[.、:：)）]\s*")


def _strip_numbering(text: str) -> str:
    """去掉"1. " "第2步：" 这类前缀。

    子任务本身已经有 id 字段，描述里再带一遍序号，拼进提示词时会出现
    "子任务目标：3. 按回车键"这种读起来别扭的东西，而且模型给的序号
    常和实际顺序对不上（截断之后尤其）。
    """
    return _NUMBER_PREFIX.sub("", text).strip()
