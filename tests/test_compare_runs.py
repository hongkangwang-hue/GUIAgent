"""对比汇总的口径测试。

## 这里守的是什么

这个脚本的输出会**直接进报告**。它算错一个分母，报告里就是一个错的
成功率，而错的成功率长得和对的一模一样——本项目已经栽过五次这种跟头
（召回率 10%→100%、OCR 86.7%→100%、模式 B 完整率 85.7%→100%、
「打开指定文件」0%→100%、稳定性「内存泄漏」）。**每一次都没有抛异常。**

所以这些用例几乎全在验分母：哪些轮次算、哪些不算。
"""

from __future__ import annotations

import json
import sys

import pytest

from scripts.compare_runs import (
    Arm,
    build_arms,
    build_parser,
    cost_note,
    failure_modes,
    parse_arms,
    per_task,
    step_latency,
)


def _round(
    task="open_browser",
    title="打开浏览器",
    attempt=1,
    verified=True,
    precondition_ok=True,
    excluded=False,
    steps=4,
    duration_s=20.0,
    loop_status="completed",
    model_said_done=True,
    cost_cny=0.0,
    latency=None,
):
    return {
        "task": task,
        "title": title,
        "attempt": attempt,
        "verified": verified,
        "precondition_ok": precondition_ok,
        "excluded": excluded,
        "steps": steps,
        "duration_s": duration_s,
        "loop_status": loop_status,
        "model_said_done": model_said_done,
        "cost_cny": cost_cny,
        "latency": latency or {"api_ms": 1000.0, "execute_ms": 100.0, "screenshot_ms": 10.0},
    }


class TestDenominator:
    """**分母口径是这个脚本唯一真正要紧的东西。**"""

    def test_无效轮不进分子也不进分母(self):
        """起点没建立是环境的失败，不是 Agent 的失败。

        当成失败会低估能力，当成成功会高估——只有排除掉这个数字才有意义。
        """
        arm = Arm(
            label="x",
            records=[
                _round(verified=True),
                _round(verified=False, precondition_ok=False),
                _round(verified=False, precondition_ok=False),
            ],
        )
        assert len(arm.valid) == 1
        assert arm.ok == 1
        assert arm.rate() == 1.0  # 1/1，不是 1/3

    def test_人为干预的轮次同样排除(self):
        arm = Arm(
            label="x",
            records=[_round(verified=True), _round(verified=False, excluded=True)],
        )
        assert len(arm.valid) == 1
        assert arm.rate() == 1.0

    def test_一轮有效的都没有时返回_None_而不是零(self):
        """**「0% 成功」和「没有数据」是两件完全不同的事。**

        用同一个 0.0 表示，报告里就分不出「跑了 25 轮全失败」和
        「一轮都没跑成」。
        """
        arm = Arm(label="x", records=[_round(precondition_ok=False)])
        assert arm.rate() is None

    def test_全部失败时返回零而不是_None(self):
        """反过来也要成立：真的 0% 就该是 0.0。"""
        arm = Arm(label="x", records=[_round(verified=False), _round(verified=False)])
        assert arm.rate() == 0.0

    def test_老存档缺字段时按有效处理(self):
        """加 `excluded` / `precondition_ok` 之前的存档没有这两个键。

        缺失按「有效」处理是对的：那时候还没有无效轮这个概念，
        所有跑过的轮次都算数。
        """
        arm = Arm(label="x", records=[{"verified": True, "steps": 2, "title": "t"}])
        assert len(arm.valid) == 1
        assert arm.rate() == 1.0


class TestPerTask:
    def test_按任务分别统计(self):
        arm = Arm(
            label="x",
            records=[
                _round(task="a", title="任务A", verified=True),
                _round(task="a", title="任务A", verified=False),
                _round(task="b", title="任务B", verified=True),
            ],
        )
        assert per_task(arm) == {"任务A": (1, 2), "任务B": (1, 1)}

    def test_无效轮不进逐任务表(self):
        arm = Arm(
            label="x",
            records=[
                _round(title="任务A", verified=True),
                _round(title="任务A", verified=False, precondition_ok=False),
            ],
        )
        assert per_task(arm) == {"任务A": (1, 1)}


class TestStepLatency:
    def test_按步平均而不是按轮平均(self):
        """**一轮 12 步和一轮 2 步不可比。**

        离线组因为卡循环把步数撑满了，按轮平均会把「每步慢」和「步数多」
        混成一个数，于是分不清延迟高是模型慢还是循环多。
        """
        arm = Arm(
            label="x",
            records=[
                _round(steps=2, latency={"api_ms": 2000.0}),
                _round(steps=8, latency={"api_ms": 8000.0}),
            ],
        )
        # 总 10000ms / 总 10 步 = 1000ms，而不是 (2000+8000)/2 = 5000
        assert step_latency(arm)["api_ms"] == pytest.approx(1000.0)

    def test_零步时不除零(self):
        arm = Arm(label="x", records=[_round(steps=0, latency={"api_ms": 500.0})])
        assert step_latency(arm)["api_ms"] is None


class TestCostNote:
    def test_离线是确认的零(self):
        """离线版的 API 费用是实打实的 0，是能写进报告的事实。"""
        arm = Arm(label="x", offline=True, records=[_round(cost_cny=0.0)])
        assert "确认值" in cost_note(arm)

    def test_在线未配单价是未知而不是零(self):
        """**两者都显示 0.0000，混为一谈就会写出「离线比在线便宜」。**

        那句话看似有据，实际上在线那一半根本没测。
        """
        arm = Arm(label="x", offline=False, records=[_round(cost_cny=0.0)])
        note = cost_note(arm)
        assert "未知" in note
        assert "确认" not in note

    def test_在线有真实成本时照实报(self):
        arm = Arm(label="x", offline=False, records=[_round(cost_cny=0.0123)])
        assert "0.0123" in cost_note(arm)


class TestFailureModes:
    def test_只统计失败的轮次(self):
        arm = Arm(
            label="x",
            records=[
                _round(verified=True, loop_status="completed"),
                _round(verified=False, loop_status="subtask_failed"),
                _round(verified=False, loop_status="plan_failed"),
            ],
        )
        modes = failure_modes(arm)
        assert modes == {"子任务失败": 1, "任务拆解失败": 1}

    def test_跑完但判定未通过单独成一类(self):
        """**这一类最要紧**：Loop 正常收尾、模型以为做完了，任务却没做成。

        它和「子任务失败」的修法完全不同，混在一起就看不出该修哪。
        """
        arm = Arm(label="x", records=[_round(verified=False, loop_status="completed")])
        assert failure_modes(arm) == {"跑完但判定未通过": 1}


class TestBuildArms:
    def _archive(self, tag, records, backend="b", offline=False):
        return {"tag": tag, "backend": backend, "offline": offline, "records": records}

    def test_多个_tag_合并成一组(self, tmp_path):
        """在线组就是两份存档：25 轮全量 + 5 轮补跑。"""
        archives = [
            (tmp_path / "a.json", self._archive("online", [_round(title="T1")])),
            (tmp_path / "b.json", self._archive("online-send", [_round(title="T2")])),
        ]
        arms = build_arms(archives, {"在线": ["online", "online-send"]})
        assert len(arms) == 1
        assert len(arms[0].records) == 2
        assert arms[0].sources == ["a.json", "b.json"]

    def test_没有_tag_的存档被跳过而不是猜(self, tmp_path, capsys):
        """**猜一个比漏掉更糟。** 归错组会让整张对比表失去意义。"""
        archives = [
            (tmp_path / "old.json", {"records": [_round()]}),
            (tmp_path / "new.json", self._archive("base", [_round()])),
        ]
        arms = build_arms(archives, None)
        assert [a.label for a in arms] == ["base"]
        assert "没有 tag" in capsys.readouterr().err

    def test_不给_mapping_时每个_tag_自成一组(self, tmp_path):
        archives = [
            (tmp_path / "a.json", self._archive("base", [_round()])),
            (tmp_path / "b.json", self._archive("lora", [_round()])),
        ]
        assert sorted(a.label for a in build_arms(archives, None)) == ["base", "lora"]

    def test_空组不出现在结果里(self, tmp_path):
        archives = [(tmp_path / "a.json", self._archive("base", [_round()]))]
        arms = build_arms(archives, {"基座": ["base"], "微调": ["lora"]})
        assert [a.label for a in arms] == ["基座"]


class TestArgs:
    def test_arm_解析(self):
        assert parse_arms(["在线=online,online-send"]) == {"在线": ["online", "online-send"]}

    def test_arm_格式错误直接报(self):
        with pytest.raises(SystemExit):
            parse_arms(["没有等号"])

    def test_不给_arm_时返回_None(self):
        assert parse_arms([]) is None

    def test_main_读到的每个属性都存在(self):
        """与 `serve_local_model` 同一条护栏，理由见那边。"""
        import inspect
        import re

        from scripts import compare_runs

        used = set(
            re.findall(r"\bargs\.([a-zA-Z_][a-zA-Z0-9_]*)", inspect.getsource(compare_runs.main))
        )
        parsed = vars(compare_runs.build_parser().parse_args([]))
        assert not used - parsed.keys()


def test_端到端跑一遍不报错(tmp_path, capsys):
    """三组齐全时能出表。**只验不炸**，数字的正确性由上面各条守。"""
    from scripts.compare_runs import render

    for tag, verified in (("online", True), ("base", False), ("lora", False)):
        payload = {
            "tag": tag,
            "backend": f"后端-{tag}",
            "offline": tag != "online",
            "records": [_round(verified=verified, attempt=i) for i in range(1, 4)],
        }
        (tmp_path / f"{tag}.json").write_text(json.dumps(payload, ensure_ascii=False), "utf-8")

    from scripts.compare_runs import load_archives

    arms = build_arms(load_archives(tmp_path), None)
    render(arms, markdown=True)
    out = capsys.readouterr().out
    assert "逐任务成功率" in out
    assert "失败形态" in out


def test_解析器有_markdown_开关():
    assert build_parser().parse_args(["--markdown"]).markdown is True


# ===================================================================== #
# 剔除标记工具
# ===================================================================== #


class TestExcludeRound:
    """`scripts/exclude_round.py` —— 唯一一个会改评测数据的工具。

    **改数据的工具必须比别的工具更严。** 这里守三条：只动该动的字段、
    理由必填、改完的成功率与 `run_basic_tasks.py` 同口径。
    """

    def _archive(self, tmp_path):
        payload = {
            "tag": "base",
            "records": [
                _round(task="a", attempt=1, verified=True),
                _round(task="a", attempt=2, verified=False),
                _round(task="b", attempt=1, verified=True),
            ],
        }
        path = tmp_path / "run.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), "utf-8")
        return path

    def test_标记之后分子分母都少一(self, tmp_path):
        from scripts.exclude_round import find, load, summarize

        path = self._archive(tmp_path)
        records = load(path)["records"]
        assert summarize(records) == "2/3 = 67%"

        find(records, "a", 1)["excluded"] = True  # 剔掉一个成功的
        assert summarize(records) == "1/2 = 50%"

    def test_理由必填(self, tmp_path):
        """**没有理由的剔除等于偷偷改数据。**

        三个月后没人说得清当时发生了什么，而「为什么少了一轮」恰恰是
        审稿人第一个会问的。
        """
        import subprocess

        path = self._archive(tmp_path)
        result = subprocess.run(
            [
                sys.executable,
                "scripts/exclude_round.py",
                str(path),
                "--task",
                "a",
                "--attempt",
                "1",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert result.returncode != 0
        assert "理由" in (result.stderr + result.stdout)

    def test_只动两个字段(self, tmp_path):
        """一个会改评测数据的工具，能改的范围越窄越好。"""
        import subprocess

        path = self._archive(tmp_path)
        before = json.loads(path.read_text(encoding="utf-8"))
        subprocess.run(
            [
                sys.executable,
                "scripts/exclude_round.py",
                str(path),
                "--task",
                "a",
                "--attempt",
                "1",
                "--reason",
                "误触鼠标",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        after = json.loads(path.read_text(encoding="utf-8"))

        changed = set()
        for old, new in zip(before["records"], after["records"], strict=True):
            changed |= {k for k in new if old.get(k) != new.get(k)}
        assert changed == {"excluded", "exclusion_reason"}

    def test_找不到的轮次报得明确(self, tmp_path):
        import subprocess

        path = self._archive(tmp_path)
        result = subprocess.run(
            [
                sys.executable,
                "scripts/exclude_round.py",
                str(path),
                "--task",
                "不存在",
                "--attempt",
                "9",
                "--reason",
                "x",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert result.returncode != 0
        assert "没有" in (result.stderr + result.stdout)

    def test_可以撤销(self, tmp_path):
        from scripts.exclude_round import find, load, summarize

        path = self._archive(tmp_path)
        records = load(path)["records"]
        record = find(records, "a", 1)
        record["excluded"] = True
        record["exclusion_reason"] = "误触"
        assert summarize(records) == "1/2 = 50%"
        record["excluded"] = False
        assert summarize(records) == "2/3 = 67%"
