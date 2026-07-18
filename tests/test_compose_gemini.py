import json
import urllib.error
from pathlib import Path

from library_learning.compose.gemini import GeminiClient, _urllib_transport, load_env_key


def test_load_env_key(tmp_path):
    env = tmp_path / ".env"
    env.write_text('GEMINI_API_KEY="bad"\nGEMINI_API_KEY_LAKELAB="AIzaGOOD"\n')
    assert load_env_key(env_path=env) == "AIzaGOOD"


def test_generate_logs_verbatim(tmp_path):
    calls = []

    def fake_transport(url, payload):
        calls.append((url, payload))
        return {"candidates": [{"content": {"parts": [{"text": "hello"}]}}],
                "modelVersion": "gemini-3.1-pro-preview"}

    client = GeminiClient(log_dir=tmp_path, api_key="k", transport=fake_transport)
    out = client.generate("say hello", tag="smoke")
    assert out == "hello"
    assert "gemini-3.1-pro-preview" in calls[0][0]
    assert calls[0][1]["generationConfig"]["temperature"] == 0.0

    log_files = sorted(tmp_path.glob("call_*.json"))
    assert len(log_files) == 1 and log_files[0].name == "call_000_smoke.json"
    logged = json.loads(log_files[0].read_text())
    assert logged["prompt"] == "say hello"
    assert logged["response"]["candidates"][0]["content"]["parts"][0]["text"] == "hello"
    assert logged["model"] == "gemini-3.1-pro-preview"

    client.generate("again", tag="smoke")
    assert (tmp_path / "call_001_smoke.json").exists()


def test_urllib_transport_retries_url_error(monkeypatch):
    import urllib.request as ur

    calls = {"n": 0}

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"ok": True}).encode()

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.URLError("connection refused")
        return FakeResp()

    monkeypatch.setattr(ur, "urlopen", fake_urlopen)
    from library_learning.compose import gemini as gemini_mod
    monkeypatch.setattr(gemini_mod.time, "sleep", lambda s: None)

    result = _urllib_transport("http://x", {"a": 1})
    assert result == {"ok": True}
    assert calls["n"] == 2


def test_log_numbering_skips_gaps(tmp_path):
    (tmp_path / "call_000_a.json").write_text("{}")
    (tmp_path / "call_005_b.json").write_text("{}")

    def fake_transport(url, payload):
        return {"candidates": [{"content": {"parts": [{"text": "x"}]}}]}

    client = GeminiClient(log_dir=tmp_path, api_key="k", transport=fake_transport)
    client.generate("p", tag="c")
    assert (tmp_path / "call_006_c.json").exists()
