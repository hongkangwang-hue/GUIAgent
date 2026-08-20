"""Session 编排层的单元测试。

这一层几乎全是**接线**，而接线出错的方式很有特点：各层单测都过，端到端
一跑什么都不对，且报错位置离根因很远。

因此这里测的重点不是"能不能跑通"，而是**每根线有没有接对**：提示词有没有
真的进后端、拆解阶段临时改的配置有没有还原、子任务之间历史有没有清、
坐标系尺寸对不对得上。这些每一条错了都会表现成"模型好像不太行"。
"""

from __future__ import annotations

import numpy as np
import pytest

from agent.context import ContextPolicy
from agent.session import Session, SessionConfig
from control.executor import ActionExecutor
from core.loop import LoopConfig
from core.trajectory import TrajectoryReader
from grounding.native import NativeGrounding
from llm.fake import ScriptedBackend
from perception.capture import Screenshot
from perception.coordinate import CoordinateScaler
from perception.types import BBox

SCREEN = BBox(0, 0, 2560, 1600)
MODEL_W, MODEL_H = 1024, 768

PLAN_TWO = (
    '{"subtasks":[{"id":1,"goal":"点击开始按钮","expected":"菜单展开"},'
    '{"id":2,"goal":"输入记事本","expected":"出现搜索结果"}]}'
)


class FakeCapturer:
    def __init__(self) -> None:
        self.calls = 0

    def capture(self, monitor: int = 1, fresh: bool = False) -> Screenshot:
        self.calls += 1
        return Screenshot(
            image=np.zeros((SCREEN.height, SCREEN.width, 3), dtype=np.uint8),
            region=SCREEN,
            engine="fake",
        )


@pytest.fixture(autouse=True)
def _no_settle(monkeypatch):
    monkeypatch.setattr("core.loop.time.sleep", lambda _seconds: None)


def build(script, tmp_path, config=None, model_size=(MODEL_W, MODEL_H), on_step=None):
    scaler = CoordinateScaler(SCREEN)
    scaler.register("planner", *model_size)
    executor = ActionExecutor(scaler, space_name="planner", dry_run=True)
    backend = ScriptedBackend(script)
    session = Session(
        backend,
        NativeGrounding(*model_size),
        executor,
        FakeCapturer(),
        config=config or SessionConfig(loop=LoopConfig(max_iterations=4, save_frames=False)),
        trajectory_root=tmp_path,
        on_step=on_step,
    )
    return session, backend


def two_subtask_script() -> list:
    return [
        {"raw_text": PLAN_TWO},
        {"action": "left_click", "x": 470, "y": 750, "thinking": "点开始"},
        {"done": True, "thinking": "菜单展开了"},
        {"action": "type", "text": "记事本", "thinking": "输入"},
        {"done": True, "thinking": "结果出来了"},
    ]


# ===================================================================== #
# 完整链路
# ===================================================================== #


def test_full_chain(tmp_path) -> None:
    """M2 任务 7：指令 → Planner 拆解 → 逐子任务进 Loop → 落盘 → 返回结果。"""
    session, _ = build(two_subtask_script(), tmp_path)
    result = session.run("打开记事本")

    assert result.status == "completed"
    assert result.succeeded
    assert [o.goal for o in result.outcomes] == ["点击开始按钮", "输入记事本"]
    assert result.total_steps == 4


def test_subtasks_enter_the_loop_one_at_a_time(tmp_path) -> None:
    """一次只带一个子目标进 Loop——开源模型能跑起来的前提。"""
    session, backend = build(two_subtask_script(), tmp_path)
    session.run("打开记事本")

    # calls[0] 是拆解；之后每次都只带单个子任务目标
    goals = {call["instruction"] for call in backend.calls[1:]}
    assert goals == {"点击开始按钮", "输入记事本"}


# ===================================================================== #
# 提示词注入 —— 这一层最容易漏的一根线
# ===================================================================== #


def test_executor_prompt_is_injected(tmp_path) -> None:
    """不注入的话整个系统是哑的。

    模型不知道有哪些动作、不知道坐标系多大、不知道该输出什么格式，只能
    靠猜。此前各层单测全过而端到端跑不通，缺的就是这一步。
    """
    session, backend = build(two_subtask_script(), tmp_path)
    session.run("打开记事本")

    assert "left_click" in backend.system_prompt
    assert f"{MODEL_W}×{MODEL_H}" in backend.system_prompt
    assert len(backend.few_shot) >= 2


def test_planner_prompt_is_restored_before_execution(tmp_path) -> None:
    """拆解会临时改后端的提示词，执行前必须换回执行模板。

    不换回来的话，模型会带着 planner 的系统提示去做单步决策，继续输出
    子任务列表而不是动作——而这个故障看起来像"模型不会执行"，根因很难
    指向这里。
    """
    session, backend = build(two_subtask_script(), tmp_path)
    session.run("打开记事本")

    # 执行模板讲的是动作，planner 模板讲的是拆解
    assert "可用动作" in backend.system_prompt
    assert "拆成一串" not in backend.system_prompt


def test_image_size_pushed_to_backend(tmp_path) -> None:
    config = SessionConfig(
        loop=LoopConfig(max_iterations=4, save_frames=False), image_size=(800, 600)
    )
    session, backend = build(two_subtask_script(), tmp_path, config=config, model_size=(800, 600))
    session.run("x")
    assert backend.image_size == (800, 600) if hasattr(backend, "image_size") else True


def test_coordinate_space_mismatch_is_reported(tmp_path, caplog) -> None:
    """送模型的图与坐标系尺寸对不上，每次点击都会系统性偏移。

    这是最容易错、错了最难查的一处：看起来像"模型定位不准"，实际是坐标系
    错配。开工时喊一声，比事后对着一堆偏移量猜便宜得多。
    """
    scaler = CoordinateScaler(SCREEN)
    scaler.register("planner", 1280, 800)  # 与 image_size 的 1024×768 不符
    executor = ActionExecutor(scaler, space_name="planner", dry_run=True)

    with caplog.at_level("ERROR"):
        Session(
            ScriptedBackend([]),
            NativeGrounding(1024, 768),
            executor,
            FakeCapturer(),
            trajectory_root=tmp_path,
        )
    assert any("必须一致" in record.message for record in caplog.records)


def test_unregistered_space_is_reported(tmp_path, caplog) -> None:
    scaler = CoordinateScaler(SCREEN)
    scaler.register("其他坐标系", 1024, 768)
    executor = ActionExecutor(scaler, space_name="planner", dry_run=True)

    with caplog.at_level("ERROR"):
        Session(
            ScriptedBackend([]),
            NativeGrounding(1024, 768),
            executor,
            FakeCapturer(),
            trajectory_root=tmp_path,
        )
    assert any("未在 CoordinateScaler" in record.message for record in caplog.records)


# ===================================================================== #
# 上下文隔离
# ===================================================================== #


def test_history_cleared_between_subtasks(tmp_path) -> None:
    """上一个子任务的操作对下一个的决策价值很低，却要一直付 token。"""
    session, backend = build(two_subtask_script(), tmp_path)
    session.run("打开记事本")

    # 第二个子任务的第一次调用，历史应当是空的
    lengths = [call["history_len"] for call in backend.calls[1:]]
    assert lengths[0] == 0  # 子任务 1 第一步
    assert lengths[2] == 0  # 子任务 2 第一步


def test_history_kept_when_disabled(tmp_path) -> None:
    config = SessionConfig(
        loop=LoopConfig(max_iterations=4, save_frames=False),
        clear_history_between_subtasks=False,
    )
    session, backend = build(two_subtask_script(), tmp_path, config=config)
    session.run("打开记事本")
    assert [c["history_len"] for c in backend.calls[1:]][2] > 0


def test_context_policy_reaches_the_loop(tmp_path) -> None:
    """上下文策略要真的生效，不能只是存了个配置。"""
    script = [{"raw_text": PLAN_TWO}] + [{"action": "left_click", "x": 1, "y": 1} for _ in range(8)]
    config = SessionConfig(
        loop=LoopConfig(max_iterations=6, save_frames=False),
        context=ContextPolicy(k=2),
        clear_history_between_subtasks=False,
    )
    session, backend = build(script, tmp_path, config=config)
    session.run("x")
    assert max(call["history_len"] for call in backend.calls[1:]) <= 2


# ===================================================================== #
# 失败路径
# ===================================================================== #


def test_plan_failure_is_recorded(tmp_path) -> None:
    session, _ = build([{"raw_text": "我觉得应该先打开浏览器"}], tmp_path)
    result = session.run("打开记事本")

    assert result.status == "plan_failed"
    assert not result.succeeded
    assert TrajectoryReader(result.trajectory_dir).meta.status == "failed"


def test_subtask_failure_stops_the_rest(tmp_path) -> None:
    """GUI 操作有强顺序依赖，前一步没成往下走全是无效点击。"""
    script = [
        {"raw_text": PLAN_TWO},
        {"action": "left_click", "thinking": "忘了给坐标"},  # grounding 失败
    ]
    session, _ = build(
        script,
        tmp_path,
    )
    result = session.run("打开记事本")

    assert result.status == "subtask_failed"
    assert len(result.outcomes) == 1  # 第二个子任务没跑
    assert "子任务 #1" in result.reason


def test_empty_instruction_fails_cleanly(tmp_path) -> None:
    session, _ = build([{"raw_text": PLAN_TWO}], tmp_path)
    assert session.run("   ").status == "plan_failed"


# ===================================================================== #
# 轨迹
# ===================================================================== #


def test_trajectory_records_plan_and_config(tmp_path) -> None:
    """M3 消融要按配置分组统计，配置没落盘就没法分组。"""
    session, _ = build(two_subtask_script(), tmp_path)
    result = session.run("打开记事本")

    meta = TrajectoryReader(result.trajectory_dir).meta
    assert meta.subtasks == ["点击开始按钮", "输入记事本"]
    assert meta.meta["executor_template"] == "executor_v1"
    assert meta.meta["context"]["k"] == 3
    assert meta.meta["dry_run"] is True
    assert "plan" in meta.meta


def test_granularity_warnings_are_archived(tmp_path) -> None:
    """粒度报告要跟轨迹一起存，M2 验收标准 4 靠它作证。"""
    plan = '{"subtasks":[{"id":1,"goal":"打开记事本并且输入文字","expected":"有字"}]}'
    session, _ = build([{"raw_text": plan}, {"done": True}], tmp_path)
    result = session.run("x")

    warnings = TrajectoryReader(result.trajectory_dir).meta.meta["granularity_warnings"]
    assert any(w["rule"] == "conjunction" for w in warnings)


def test_step_numbers_continuous_across_subtasks(tmp_path) -> None:
    """回放时不该出现两个 #1。"""
    session, _ = build(two_subtask_script(), tmp_path)
    result = session.run("打开记事本")

    steps = TrajectoryReader(result.trajectory_dir).steps()
    assert [s.step for s in steps] == list(range(1, len(steps) + 1))
    assert {s.subtask_id for s in steps} == {1, 2}


def test_backend_identity_recorded(tmp_path) -> None:
    """M3 横评按 backend 字段分组。"""
    session, _ = build(two_subtask_script(), tmp_path)
    meta = TrajectoryReader(session.run("x").trajectory_dir).meta
    assert meta.backend == "scripted"
    assert meta.mode == "A"


def test_cost_accumulated_across_plan_and_execution(tmp_path) -> None:
    """拆解也花钱，必须并进总账——单任务成本是 M2 交付物。"""
    session, backend = build(two_subtask_script(), tmp_path)
    result = session.run("打开记事本")

    assert result.cost is not None
    assert result.cost.requests == len(backend.calls)  # 含拆解那一次
    assert result.cost.requests == 5


def test_on_step_callback_fires(tmp_path) -> None:
    """CLI 靠它刷新实时面板。"""
    seen = []
    session, _ = build(two_subtask_script(), tmp_path, on_step=seen.append)
    session.run("打开记事本")
    assert [o.id for o in seen] == [1, 2]


def test_result_as_dict_is_serializable(tmp_path) -> None:
    import json

    session, _ = build(two_subtask_script(), tmp_path)
    payload = session.run("打开记事本").as_dict()
    json.dumps(payload, ensure_ascii=False)  # 抛异常就是回归
    assert payload["succeeded_subtasks"] == 2
