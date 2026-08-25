"""重试探针的行为测试。

## 这里守的是两件事

**一、判定要对。** 探针回答的是「重试时模型有没有改主意」。判错了，
M4 任务 2 会照着错误的前提设计——重复率虚低会让人以为「加重试次数就行」。

**二、不能把客机屏幕上的东西带出来。** 轨迹里有整屏截图，`model_thinking`
是模型对屏幕的描述。M2 记录过两次真实事故（Agent 改写并保存了 `.env`、
对着真实微软账号登录框按了 12 次回车）——**那两次的截图里就有这些**。
探针的产物是要提交进仓库的，默认必须只导结构化数据。
"""

from __future__ import annotations

import json

import pytest

from scripts.probe_retries import (
    SAME_SPOT_RADIUS,
    analyse,
    analyse_subtask,
    same_attempt,
    summarize,
)


def step(subtask_id=1, action="left_click", x=470, y=750, status="ok", **extra):
    row = {
        "subtask_id": subtask_id,
        "subtask": "点击任务栏的开始按钮",
        "execution_status": status,
        "action_intent": {"action_type": action, "params": {"x": x, "y": y}},
        "action_model_coords": {"action": action, "x": x, "y": y},
        "model_thinking": "任务栏上的开始按钮位于(470, 750)",
        "screenshot_before": "frames/step001-before.png",
    }
    row.update(extra)
    return row


class TestSameAttempt:
    def test_逐字相同算同一次(self):
        assert same_attempt(("left_click", 470, 750), ("left_click", 470, 750))

    def test_差几个像素也算同一次(self):
        """**GUI 按钮几十像素宽。** 只按逐字相等算会低估重复率——

        模型每次抖动两三个像素，看起来像「在改主意」，其实点的是同一个东西。
        """
        assert same_attempt(("left_click", 470, 750), ("left_click", 473, 752))

    def test_差得远不算同一次(self):
        assert not same_attempt(("left_click", 470, 750), ("left_click", 470, 900))

    def test_动作类型不同就不算(self):
        """坐标一样但一个点击一个双击，那是真的换了策略。"""
        assert not same_attempt(("left_click", 470, 750), ("double_click", 470, 750))

    def test_半径边界(self):
        assert same_attempt(("a", 0, 0), ("a", SAME_SPOT_RADIUS, 0))
        assert not same_attempt(("a", 0, 0), ("a", SAME_SPOT_RADIUS + 1, 0))

    def test_都没有坐标时算同一次(self):
        """`done`、按键这类动作没有坐标，两次都没有就是同一次。"""
        assert same_attempt(("key", None, None), ("key", None, None))

    def test_一个有坐标一个没有不算同一次(self):
        assert not same_attempt(("left_click", 470, 750), ("left_click", None, None))


class TestRepeatRate:
    def test_全部重复时重复率为一(self):
        """**这是要抓的那个形态**：六步全在点同一个地方。"""
        row = analyse_subtask([step() for _ in range(6)], include_text=False)
        assert row["distinct_attempts"] == 1
        assert row["max_repeat"] == 6

    def test_每次都不同时没有重复(self):
        steps = [step(x=100 * i, y=100 * i) for i in range(1, 7)]
        row = analyse_subtask(steps, include_text=False)
        assert row["distinct_attempts"] == 6
        assert row["max_repeat"] == 1

    def test_重复率算的是尝试不是步数(self, tmp_path):
        traj = tmp_path / "traj-x"
        traj.mkdir()
        steps = [step() for _ in range(4)] + [step(x=200, y=200)]
        traj.joinpath("steps.jsonl").write_text(
            "\n".join(json.dumps(s, ensure_ascii=False) for s in steps), encoding="utf-8"
        )
        out = analyse(traj, include_text=False)
        assert out["steps"] == 5
        assert out["distinct_attempts"] == 2
        assert out["repeat_rate"] == pytest.approx(0.6)

    def test_子任务分开统计(self, tmp_path):
        """两个子任务各点各的，不能算成互相重复。"""
        traj = tmp_path / "traj-y"
        traj.mkdir()
        steps = [step(subtask_id=1) for _ in range(2)] + [
            step(subtask_id=2, x=900, y=100) for _ in range(2)
        ]
        traj.joinpath("steps.jsonl").write_text(
            "\n".join(json.dumps(s, ensure_ascii=False) for s in steps), encoding="utf-8"
        )
        out = analyse(traj, include_text=False)
        assert out["subtasks"] == 2
        assert out["distinct_attempts"] == 2

    def test_坏行跳过而不是整份作废(self, tmp_path):
        """断电时最后一行天然可能残缺。"""
        traj = tmp_path / "traj-z"
        traj.mkdir()
        traj.joinpath("steps.jsonl").write_text(
            json.dumps(step(), ensure_ascii=False) + '\n{"subtask_id": 1, "act', encoding="utf-8"
        )
        assert analyse(traj, include_text=False)["steps"] == 1

    def test_没有_steps_文件时返回_None(self, tmp_path):
        traj = tmp_path / "empty"
        traj.mkdir()
        assert analyse(traj, include_text=False) is None


class TestSummary:
    def test_点出每次尝试都一样的子任务数(self):
        runs = [
            {
                "steps": 10,
                "distinct_attempts": 3,
                "detail": [
                    {"steps": 6, "distinct_attempts": 1},
                    {"steps": 3, "distinct_attempts": 1},
                    {"steps": 1, "distinct_attempts": 1},
                ],
            }
        ]
        stats = summarize(runs)
        # 只算「有重试的」子任务：1 步的那个没重试过，不该进分母
        assert stats["subtasks_with_retries"] == 2
        assert stats["subtasks_where_every_retry_identical"] == 2

    def test_单步子任务不算有重试(self):
        runs = [
            {
                "steps": 2,
                "distinct_attempts": 2,
                "detail": [{"steps": 1, "distinct_attempts": 1}] * 2,
            }
        ]
        assert summarize(runs)["subtasks_with_retries"] == 0


class TestDataBoundary:
    """**产物要提交进仓库，不能带出客机屏幕上的东西。**"""

    def test_默认不导出截图路径(self):
        row = analyse_subtask([step()], include_text=False)
        assert "frames" not in json.dumps(row, ensure_ascii=False)

    def test_默认不导出模型对屏幕的描述(self):
        """`model_thinking` 是模型在描述屏幕内容。

        M2 那次 Agent 打开 `.env` 时，屏幕上就是凭据。
        """
        row = analyse_subtask([step()], include_text=False)
        blob = json.dumps(row, ensure_ascii=False)
        assert "任务栏上的开始按钮" not in blob
        assert "点击任务栏的开始按钮" not in blob

    def test_显式打开时才导出且截断(self):
        long_text = "机密" * 200
        row = analyse_subtask([step(model_thinking=long_text)], include_text=True)
        from scripts.probe_retries import TEXT_CAP

        assert len(row["thinking"][0]) <= TEXT_CAP

    def test_结构化字段始终在(self):
        """把「不导文本」和「不导数据」分清楚——动作和坐标必须留着。"""
        row = analyse_subtask([step()], include_text=False)
        assert row["attempts"] == [{"action": "left_click", "x": 470, "y": 750}]
        assert row["statuses"] == {"ok": 1}


class TestCli:
    def test_参数表不漏接(self):
        import ast
        import inspect

        from scripts import probe_retries

        tree = ast.parse(inspect.getsource(probe_retries))
        used = {
            n.attr
            for n in ast.walk(tree)
            if isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name)
            and n.value.id == "args"
        }
        parsed = vars(probe_retries.build_parser().parse_args([]))
        assert used and not used - parsed.keys()

    def test_默认不导文本(self):
        from scripts.probe_retries import build_parser

        assert build_parser().parse_args([]).include_text is False
