"""Reflector 的行为测试 —— M4 任务 1。

每条设计选择都钉一个用例，尤其是**两条刻意的保守选择**：
不否决「0 动作就 done」、否决次数有上限。这两条如果被后来的人
「顺手优化」掉，会分别制造出新的失败模式和死循环。
"""

from __future__ import annotations

import pytest

from core.loop import LoopConfig
from core.reflector import (
    VERDICT_ACCEPT,
    VERDICT_REJECT,
    VERDICT_UNKNOWN,
    Reflector,
)


class TestReflector级联1:
    def test_动作后屏幕变了就接受(self):
        r = Reflector()
        r.observe(changed=True)
        v = r.judge()
        assert v.verdict == VERDICT_ACCEPT
        assert v.level == 1
        assert v.hint == ""

    def test_动作后屏幕没变就否决(self):
        """这是本模块存在的理由。

        实测：动作 execution_status=ok 之后立刻报 done 有 78 次，
        而 21/25 轮以 done 收尾却判定失败。
        """
        r = Reflector()
        r.observe(changed=False)
        v = r.judge()
        assert v.verdict == VERDICT_REJECT
        assert v.level == 1
        assert v.hint, "否决必须带反馈，否则模型下一轮还给同一个答案"

    def test_否决的反馈要说清没生效而不只是再试一次(self):
        """`core/retry.py` 实测：只升级动作类型，坐标重复率仍有 75.3%。

        要让模型换目标，反馈里必须出现「没有变化 / 没有命中」这层意思。
        """
        r = Reflector()
        r.observe(changed=False)
        hint = r.judge().hint
        assert "没有发生变化" in hint or "没有命中" in hint
        assert "不要报告完成" in hint


class TestReflector保守选择:
    def test_零动作就done不否决只标记(self):
        """实测 12 个子任务（13.3%）属于这种。

        它可能是合法的——子任务目标本来就已满足。M2 那批假成功正是
        precondition 没重置好、判据一开始就满足。现有数据分不开
        「合法的已满足」和「纯粹偷懒」，**贸然否决会制造新的失败模式**。
        """
        r = Reflector()
        v = r.judge()
        assert v.verdict == VERDICT_UNKNOWN
        assert not v.rejected
        assert v.hint == ""

    def test_缺帧差数据时不否决(self):
        """没有帧差就没有依据，没有依据不能否决。"""
        r = Reflector()
        r.observe(changed=None)
        v = r.judge()
        assert v.verdict == VERDICT_UNKNOWN
        assert not v.rejected

    def test_连续否决有上限否则死循环(self):
        """否决之后模型可能原样再报一次 done。没有上限就是死循环。"""
        r = Reflector(max_rejects=2)
        r.observe(changed=False)
        assert r.judge().verdict == VERDICT_REJECT
        assert r.judge().verdict == VERDICT_REJECT
        v = r.judge()
        assert v.verdict == VERDICT_ACCEPT, "到上限必须接受，交程序化判定兜底"
        assert "兜底" in v.reason

    def test_上限可配(self):
        r = Reflector(max_rejects=1)
        r.observe(changed=False)
        assert r.judge().verdict == VERDICT_REJECT
        assert r.judge().verdict == VERDICT_ACCEPT


class TestReflector状态:
    def test_reset清空全部状态(self):
        r = Reflector()
        r.observe(changed=False)
        r.judge()
        assert r.rejects == 1 and r.actions_taken == 1
        r.reset()
        assert r.rejects == 0 and r.actions_taken == 0 and r.last_changed is None

    def test_judge只在否决时改状态(self):
        """接受路径必须可重复调用而不累积计数。"""
        r = Reflector()
        r.observe(changed=True)
        for _ in range(5):
            assert r.judge().verdict == VERDICT_ACCEPT
        assert r.rejects == 0

    def test_多次observe以最后一次为准(self):
        r = Reflector()
        r.observe(changed=False)
        r.observe(changed=True)
        assert r.actions_taken == 2
        assert r.judge().verdict == VERDICT_ACCEPT


class TestLoopConfig接线:
    def test_默认关闭(self):
        """**M2/M3 的全部实测都是在关闭下跑的。**

        默认打开等于让那些复现命令悄悄跑出另一套结果。
        """
        assert LoopConfig().reflector is False

    def test_打开后帧差条件也成立(self):
        """级联 1 就是帧差。没有帧差 Reflector 只能一路 unknown，等于没开。"""
        import inspect

        from core import loop as loop_mod

        src = inspect.getsource(loop_mod)
        assert "self.config.escalate_on_no_change or self.config.reflector" in src

    def test_否决路径不结束子任务(self):
        """否决必须 `return ..., None`，让循环进下一轮，而不是返回 STOP_DONE。"""
        import inspect

        from core import loop as loop_mod

        src = inspect.getsource(loop_mod)
        i = src.find("if verdict.rejected:")
        assert i > 0, "找不到否决分支"
        tail = src[i : i + 600]
        assert "return self._commit(record), None" in tail

    @pytest.mark.parametrize("field", ["reflector", "reflector_max_rejects"])
    def test_配置项存在(self, field):
        assert hasattr(LoopConfig(), field)
