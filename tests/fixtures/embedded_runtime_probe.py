"""Run installed or editable public APIs against a loopback model replay."""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main():
    root = Path(os.environ["HOME"])
    before = dict(os.environ)
    from run_agent import AIAgent
    from hermes_runtime import embedded_runtime
    from hermes_constants import get_hermes_home
    from hermes_cli.config import load_config
    from agent.secret_scope import get_secret

    assert dict(os.environ) == before
    assert not Path(os.environ["HERMES_HOME"]).exists()
    requests = []

    class Replay(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path != "/v1/chat/completions":
                self.send_error(404)
                return
            requests.append(payload)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for delta, reason in [
                ({"role": "assistant", "content": "Replay answer"}, None),
                ({}, "stop"),
            ]:
                chunk = {
                    "id": "replay",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "embedded-replay",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": reason}],
                }
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Replay)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    homes = []
    try:
        for name in ("a", "b", "a"):
            home = root / name
            home.mkdir(exist_ok=True)
            # A conflicting on-disk config must never replace the supplied snapshot.
            (home / "config.yaml").write_text(
                "model:\n  default: disk-model\n", encoding="utf-8"
            )
            initial = (home / "config.yaml").read_bytes()
            home.chmod(0o500)
            history = []
            deltas = []
            start = len(requests)
            try:
                with embedded_runtime(
                    config={
                        "model": {"default": name, "context_length": 131072},
                        "agent": {"environment_probe": False},
                        "compression": {"enabled": False},
                    },
                    config_home=home,
                    scratch_dir=root / (name + "-scratch"),
                ):
                    assert get_hermes_home() == home.resolve()
                    assert load_config()["model"]["default"] == name
                    assert get_secret("OPENAI_API_KEY") is None
                    for turn in range(2):
                        agent = AIAgent(
                            model="embedded-replay",
                            provider="openai",
                            api_mode="chat_completions",
                            api_key="synthetic-key",
                            base_url=f"http://127.0.0.1:{server.server_port}/v1",
                            session_id=f"replay-{name}",
                            session_db=None,
                            enabled_toolsets=[],
                            skip_context_files=True,
                            load_soul_identity=False,
                            quiet_mode=True,
                            run_budget_seconds=10,
                            max_iterations=2,
                            stream_delta_callback=deltas.append,
                        )
                        try:
                            result = agent.run_conversation(
                                f"Synthetic question {turn}",
                                system_message="Frozen replay instructions",
                                conversation_history=history,
                            )
                            assert result["final_response"] == "Replay answer", result
                            history = result["messages"]
                        finally:
                            agent.close()
                    assert deltas
                    assert [message["role"] for message in history] == [
                        "user",
                        "assistant",
                        "user",
                        "assistant",
                    ]
                    systems = [
                        [m["content"] for m in r["messages"] if m["role"] == "system"]
                        for r in requests[start:]
                    ]
                    assert len(systems) == 2 and systems[0] and systems[0] == systems[1]
                    assert systems[0] == ["Frozen replay instructions"]
            finally:
                home.chmod(0o700)
            assert sorted(path.name for path in home.iterdir()) == ["config.yaml"]
            assert (home / "config.yaml").read_bytes() == initial
            assert dict(os.environ) == before
            homes.append(name)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert not Path(os.environ["HERMES_HOME"]).exists()
    assert not list(root.rglob("*.db"))
    print(
        json.dumps({
            "homes": homes,
            "turns": len(requests),
            "environment_unchanged": True,
        })
    )


if __name__ == "__main__":
    main()
