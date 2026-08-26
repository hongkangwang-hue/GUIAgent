"""动作定义、参数校验与 JSON Schema 生成的测试。"""

from __future__ import annotations

import json

import pytest

from control.actions import (
    ACTION_SPECS,
    CORE_ACTIONS,
    STUB_ACTIONS,
    Action,
    ActionType,
    ActionValidationError,
    action_schema,
    to_tool_schemas,
    to_unified_tool_schema,
)

# --------------------------------------------------------------------- #
# 构造与校验
# --------------------------------------------------------------------- #


def test_click_requires_coordinates() -> None:
    with pytest.raises(ActionValidationError, match="缺少必填参数"):
        Action(ActionType.LEFT_CLICK)


def test_click_with_coordinates_is_valid() -> None:
    action = Action(ActionType.LEFT_CLICK, x=100, y=200)
    assert action.requires_coordinates()


def test_screenshot_needs_no_params() -> None:
    assert Action(ActionType.SCREENSHOT).requires_coordinates() is False


def test_unknown_action_type_rejected() -> None:
    with pytest.raises(ActionValidationError, match="未知动作类型"):
        Action("teleport")  # type: ignore[arg-type]


def test_string_action_type_is_coerced() -> None:
    assert Action("left_click", x=1, y=2).type is ActionType.LEFT_CLICK


def test_drag_requires_both_endpoints() -> None:
    with pytest.raises(ActionValidationError):
        Action(ActionType.LEFT_CLICK_DRAG, x=0, y=0, to_x=10)


def test_scroll_direction_must_be_in_enum() -> None:
    with pytest.raises(ActionValidationError, match="不在允许值"):
        Action(ActionType.SCROLL, x=0, y=0, direction="diagonal")


def test_scroll_amount_is_optional() -> None:
    Action(ActionType.SCROLL, x=0, y=0, direction="down")


def test_scroll_amount_respects_bounds() -> None:
    with pytest.raises(ActionValidationError, match="大于上限"):
        Action(ActionType.SCROLL, x=0, y=0, direction="down", amount=999)


def test_wait_duration_bounds() -> None:
    Action(ActionType.WAIT, duration=1.0)
    with pytest.raises(ActionValidationError, match="小于下限"):
        Action(ActionType.WAIT, duration=0.0)


# --------------------------------------------------------------------- #
# 序列化
# --------------------------------------------------------------------- #


def test_to_dict_only_includes_used_params() -> None:
    """轨迹日志里不该出现一堆 null——那会让 JSONL 难读且体积翻倍。"""
    payload = Action(ActionType.LEFT_CLICK, x=1, y=2).to_dict()
    assert payload == {"action": "left_click", "x": 1, "y": 2}


def test_roundtrip_through_dict() -> None:
    original = Action(ActionType.SCROLL, x=10, y=20, direction="down", amount=5)
    assert Action.from_dict(original.to_dict()).to_dict() == original.to_dict()


def test_from_dict_accepts_type_key() -> None:
    """不同模型输出习惯不同，action / type 两种键名都要吃得下。"""
    assert Action.from_dict({"type": "left_click", "x": 1, "y": 2}).type is ActionType.LEFT_CLICK


def test_from_dict_missing_action_key() -> None:
    with pytest.raises(ActionValidationError, match="缺少 action"):
        Action.from_dict({"x": 1, "y": 2})


def test_from_dict_keeps_unknown_fields_in_meta() -> None:
    """模型多输出的字段不该让解析失败，但也不能悄悄丢掉——留在 meta 里备查。"""
    action = Action.from_dict({"action": "left_click", "x": 1, "y": 2, "element_id": 7})
    assert action.meta["element_id"] == 7


def test_reasoning_is_preserved() -> None:
    action = Action(ActionType.LEFT_CLICK, x=1, y=2, reasoning="点击搜索按钮")
    assert action.to_dict()["reasoning"] == "点击搜索按钮"


# --------------------------------------------------------------------- #
# JSON Schema
# --------------------------------------------------------------------- #


def test_every_action_has_a_spec() -> None:
    """漏定义一个动作，它就永远不会出现在给模型的工具列表里。"""
    assert set(ACTION_SPECS) == set(ActionType)


def test_core_and_stub_partition_all_actions() -> None:
    assert set(ActionType) == CORE_ACTIONS | STUB_ACTIONS
    assert not (CORE_ACTIONS & STUB_ACTIONS)


def test_schema_is_json_serializable() -> None:
    json.dumps(to_tool_schemas(), ensure_ascii=False)
    json.dumps(to_unified_tool_schema(), ensure_ascii=False)


def test_tool_schemas_cover_core_actions_only_by_default() -> None:
    names = {tool["name"] for tool in to_tool_schemas()}
    assert names == {a.value for a in CORE_ACTIONS}


def test_tool_schemas_can_include_stubs() -> None:
    names = {tool["name"] for tool in to_tool_schemas(only_core=False)}
    assert "middle_click" in names


def test_every_param_has_description() -> None:
    """参数描述是开源模型能否用对工具的关键，不能有空描述。"""
    for tool in to_tool_schemas(only_core=False):
        for name, prop in tool["parameters"]["properties"].items():
            assert prop.get("description"), f"{tool['name']}.{name} 缺少描述"


def test_unified_schema_has_action_enum() -> None:
    schema = to_unified_tool_schema()
    action_prop = schema["parameters"]["properties"]["action"]
    assert set(action_prop["enum"]) == {a.value for a in CORE_ACTIONS}
    assert schema["parameters"]["required"] == ["action"]


def test_unified_schema_documents_params_per_action() -> None:
    """扁平参数表下，"哪个动作要哪些参数"只能靠 action 的描述传达。"""
    description = to_unified_tool_schema()["parameters"]["properties"]["action"]["description"]
    for action_type in CORE_ACTIONS:
        assert action_type.value in description


def test_unified_schema_merges_all_params() -> None:
    properties = to_unified_tool_schema()["parameters"]["properties"]
    for name in ("x", "y", "to_x", "to_y", "text", "keys", "direction", "amount", "duration"):
        assert name in properties


def test_schema_validation_and_runtime_validation_agree() -> None:
    """schema 里标 required 的参数，构造时不给必须报错。

    这两处一旦分家，模型就会按 schema 给参数、被运行时拒绝，或者反过来。
    """
    for action_type in CORE_ACTIONS:
        schema = action_schema(action_type)
        required = schema["parameters"]["required"]
        if not required:
            continue
        with pytest.raises(ActionValidationError):
            Action(action_type)
