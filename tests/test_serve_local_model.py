"""本地模型服务的协议测试。

## 这里守的是什么

M0 定的拓扑是「模型在宿主机 GPU 上，客机经 HTTP 调用 OpenAI 兼容端点」。
成立的前提是**服务端说的协议与客机侧 `OpenAICompatBackend` 说的是同一个**
——不一致的话，症状是客机报一个指向解析层的错，而根因在服务端，
排查要横跨两台机器。

所以这些用例基本都在验一件事：客机发出去的东西，服务端能原样理解。
全部用假后端，不加载权重、不碰 GPU。
"""

from __future__ import annotations

import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from llm.base import TokenUsage
from scripts.serve_local_model import build_app, to_hf_messages


class _FakeBackend:
    """记下收到了什么，返回一个固定答案。"""

    model = "fake/model"
    adapter = ""

    def __init__(self):
        self._model = None
        self.seen: list[tuple] = []

    def generate(self, messages, images=None, max_new_tokens=0):
        self.seen.append((messages, list(images or []), max_new_tokens))
        return (
            '{"action": "left_click", "x": 12, "y": 34}',
            TokenUsage(prompt_tokens=1200, completion_tokens=20),
            123.4,
        )


@pytest.fixture
def backend():
    return _FakeBackend()


@pytest.fixture
def client(backend):
    return TestClient(build_app(backend))


def _data_url(width: int = 8, height: int = 6) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (10, 20, 30)).save(buffer, format="JPEG")
    payload = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{payload}"


class TestMessageConversion:
    """OpenAI 的 messages → HF chat 格式。"""

    def test_纯文本消息(self):
        messages, images = to_hf_messages([{"role": "system", "content": "你是一个智能体"}])
        assert messages == [
            {"role": "system", "content": [{"type": "text", "text": "你是一个智能体"}]}
        ]
        assert images == []

    def test_图片被解出来并留下占位符(self):
        raw = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "看这张图"},
                    {"type": "image_url", "image_url": {"url": _data_url()}},
                ],
            }
        ]
        messages, images = to_hf_messages(raw)
        kinds = [part["type"] for part in messages[0]["content"]]
        assert kinds == ["text", "image"]
        assert len(images) == 1
        assert images[0].size == (8, 6)

    def test_多张图的顺序被保持(self):
        """**顺序错开一位，模型看到的就是上一帧。**

        processor 按占位符出现的先后去 images 列表里取。这种错不报错，
        只会让每一步都基于过期画面决策，症状是「模型总慢半拍」。
        """
        raw = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _data_url(4, 4)}},
                    {"type": "text", "text": "旧帧"},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _data_url(9, 9)}},
                    {"type": "text", "text": "当前"},
                ],
            },
        ]
        _, images = to_hf_messages(raw)
        assert [img.size for img in images] == [(4, 4), (9, 9)]

    def test_拒绝外部链接的图片(self):
        """认了 http 链接就等于让本服务去访问外网，

        而这个服务存在的全部理由是数据不出本机。
        """
        raw = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}
                ],
            }
        ]
        with pytest.raises(ValueError, match="data URL"):
            to_hf_messages(raw)

    def test_坏掉的_base64_报得明确(self):
        raw = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,@@"}}
                ],
            }
        ]
        with pytest.raises(ValueError, match="base64"):
            to_hf_messages(raw)


class TestChatCompletions:
    def test_返回_openai_形状的响应(self, client):
        response = client.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": [{"role": "user", "content": "点哪"}]},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert body["choices"][0]["message"]["content"].startswith("{")
        assert body["choices"][0]["finish_reason"] == "stop"

    def test_用量原样回传(self, client):
        """客机的成本统计靠它。服务端自己不记账——那次调用属于客机。"""
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "x"}]},
        )
        usage = response.json()["usage"]
        assert usage == {"prompt_tokens": 1200, "completion_tokens": 20, "total_tokens": 1220}

    def test_max_tokens_传到后端(self, client, backend):
        client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "x"}], "max_tokens": 64},
        )
        assert backend.seen[-1][2] == 64

    def test_图片送到后端(self, client, backend):
        client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "看图"},
                            {"type": "image_url", "image_url": {"url": _data_url()}},
                        ],
                    }
                ]
            },
        )
        _, images, _ = backend.seen[-1]
        assert len(images) == 1

    def test_空_messages_报_400(self, client):
        response = client.post("/v1/chat/completions", json={"messages": []})
        assert response.status_code == 400

    def test_明确拒绝流式(self, client):
        """假装支持的后果：客户端等一个 SSE 流却收到一个 JSON，

        报出来的错指向解析层，查半天才发现是服务端没实现。
        """
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "x"}], "stream": True},
        )
        assert response.status_code == 400
        assert "stream" in response.json()["detail"]

    def test_坏图片报_400_而不是_500(self, client):
        """客户端发错东西是 4xx。报 500 会让人去查服务端的日志，白费功夫。"""
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": "https://x.com/a.png"}}
                        ],
                    }
                ]
            },
        )
        assert response.status_code == 400

    def test_推理异常返回_500_且带原因(self, client, backend):
        def boom(*_args, **_kwargs):
            raise RuntimeError("CUDA out of memory")

        backend.generate = boom
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "x"}]},
        )
        assert response.status_code == 500
        assert "out of memory" in response.json()["error"]["message"]


class TestProbes:
    def test_health_区分服务在不在与权重加载没加载(self, client):
        """客机连不上时先打这个。

        「服务没起」和「模型还在加载」是两种完全不同的等待，混在一起
        只能靠猜。
        """
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["loaded"] is False

    def test_models_端点存在(self, client):
        """有些客户端启动时会探它，缺了会报一个很难懂的错。"""
        body = client.get("/v1/models").json()
        assert body["data"][0]["id"] == "fake/model"


class TestProviderEntry:
    def test_selfhost_已注册且指向本机(self, monkeypatch):
        from llm.providers import resolve

        monkeypatch.setenv("SELFHOST_API_KEY", "local")
        config = resolve("selfhost")
        assert config.base_url.startswith("http://127.0.0.1")

    def test_自建服务不需要配_api_key(self, monkeypatch):
        """本服务不校验鉴权，就别逼人去 .env 里填一个假值。

        客机上少一步手工编辑 `.env`，就少一处能改错、能存错编码的地方。
        """
        from llm.providers import resolve

        monkeypatch.delenv("SELFHOST_API_KEY", raising=False)
        config = resolve("selfhost")
        assert config.api_key  # 非空占位符，否则 ChatOpenAI 构造时就报错

    def test_云平台缺_key_仍然报错(self, monkeypatch):
        """免填 key 是给自建服务开的口子，不能把凭据校验整个废掉。"""
        from llm.providers import ProviderNotConfigured, resolve

        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        with pytest.raises(ProviderNotConfigured):
            resolve("dashscope")

    def test_base_url_可以由命令行覆盖(self, monkeypatch):
        """宿主机 IP 随机器变，写死在 .env 里不如命令行给。"""
        from llm.factory import build_backend

        monkeypatch.delenv("SELFHOST_BASE_URL", raising=False)
        backend = build_backend(provider="selfhost", base_url="http://10.0.0.5:9000/v1")
        assert backend.config.base_url == "http://10.0.0.5:9000/v1"

    def test_selfhost_被判为离线(self, monkeypatch):
        """**它和连百炼用的是同一个后端类**，光看类型分不出来。

        漏判的后果很具体：客机跑完一整轮离线实验，存档标成 `-online`，
        还顺手覆盖掉 M2 的在线交付数据。
        """
        from llm.factory import build_backend, describe, is_offline

        monkeypatch.setenv("SELFHOST_API_KEY", "local")
        backend = build_backend(provider="selfhost")
        assert is_offline(backend)
        assert "离线" in describe(backend)

    def test_云平台不被判为离线(self, monkeypatch):
        from llm.factory import build_backend, describe, is_offline

        monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
        backend = build_backend(provider="dashscope")
        assert not is_offline(backend)
        assert "在线" in describe(backend)

    def test_自建服务的零成本是确认的零(self, monkeypatch):
        """`priced=True` 才说明这个 0 可以写进报告。

        未配单价的商业平台也显示 0.0000，但那是「不知道」。两者都是 0，
        只有 priced 分得清。口径与 `llm.qwen_vl_local.LOCAL_PRICE` 一致。
        """
        from llm.providers import resolve

        monkeypatch.setenv("SELFHOST_API_KEY", "local")
        config = resolve("selfhost")
        assert config.price is not None
        assert config.price.cost_of(10_000, 5_000) == 0.0

    def test_商业平台未配单价时仍然是未知(self, monkeypatch):
        """`weights_local` 不能顺手把「不猜单价」这条规矩废掉。"""
        from llm.providers import resolve

        monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
        monkeypatch.delenv("DASHSCOPE_PRICE_IN_PER_1K", raising=False)
        monkeypatch.delenv("DASHSCOPE_PRICE_OUT_PER_1K", raising=False)
        assert resolve("dashscope").price is None
