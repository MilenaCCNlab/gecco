"""Logged Gemini REST client (reproducibility: every prompt/response on disk)."""
import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from ..config import REPO_ROOT

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_MODEL = "gemini-3.1-pro-preview"
ENV_VAR = "GEMINI_API_KEY_LAKELAB"  # plain GEMINI_API_KEY in .env is invalid
RETRY_STATUSES = {429, 500, 503}


def load_env_key(env_path=None, var=ENV_VAR):
    env_path = Path(env_path) if env_path else REPO_ROOT / ".env"
    for line in env_path.read_text().splitlines():
        m = re.match(r'\s*%s\s*=\s*"?([^"\s]+)"?' % re.escape(var), line)
        if m:
            return m.group(1)
    raise KeyError("%s not found in %s" % (var, env_path))


def _urllib_transport(url, payload):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    last_err = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code not in RETRY_STATUSES:
                raise RuntimeError("Gemini HTTP %s: %s" % (e.code, e.read()[:500]))
            time.sleep(2 ** attempt)
    raise RuntimeError("Gemini API failed after 5 retries: %s" % last_err)


class GeminiClient:
    def __init__(self, log_dir, api_key=None, model=DEFAULT_MODEL,
                 temperature=0.0, transport=None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key or load_env_key()
        self.model = model
        self.temperature = temperature
        self.transport = transport or _urllib_transport
        self._n = len(list(self.log_dir.glob("call_*.json")))

    def generate(self, prompt, tag):
        url = API_URL.format(model=self.model) + "?key=" + self.api_key
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": self.temperature},
        }
        response = self.transport(url, payload)
        record = {
            "model": self.model,
            "modelVersion": response.get("modelVersion"),
            "generationConfig": payload["generationConfig"],
            "prompt": prompt,
            "response": response,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        path = self.log_dir / ("call_%03d_%s.json" % (self._n, tag))
        path.write_text(json.dumps(record, indent=2))
        self._n += 1
        return response["candidates"][0]["content"]["parts"][0]["text"]
