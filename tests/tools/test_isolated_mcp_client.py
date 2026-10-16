"""Public isolated MCP contracts against real SDK transports and local servers.

The fixtures speak JSON-RPC on actual subprocess pipes and loopback sockets. They
deliberately do not substitute ClientSession, a transport, or native MCP tasks.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import socket
import sys
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit

import psutil
import pytest

from agent.secret_scope import reset_secret_scope, set_secret_scope
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools.mcp_isolated import IsolatedMCPClient


_STDIO_SERVER = r'''
import json
import os
import sys
import threading
import time
from pathlib import Path

config = json.loads(sys.argv[1])
mode = config.get("mode", "normal")
phase = "initial"
write_lock = threading.Lock()
launch = 1
if config.get("pid_file"):
    pid_file = Path(config["pid_file"])
    previous = pid_file.read_text().splitlines() if pid_file.exists() else []
    launch = len(previous) + 1
    with pid_file.open("a") as stream:
        stream.write(str(os.getpid()) + "\n")

def send(value):
    with write_lock:
        print(json.dumps(value), flush=True)

def result(request, value):
    send({"jsonrpc": "2.0", "id": request["id"], "result": value})

def tools():
    name = "after_refresh" if phase == "refreshed" else "echo"
    if mode == "reconnect" and launch > 1:
        name = "after_reconnect"
    return [{"name": name, "description": "Return arguments and the child profile.",
             "inputSchema": {"type": "object", "properties": {
                 "nonce": {"type": "string"}}, "additionalProperties": True}}]

if mode == "private":
    print(os.environ["FIXTURE_STDERR_SECRET"], file=sys.stderr, flush=True)

for line in sys.stdin:
    request = json.loads(line)
    method = request.get("method")
    if method == "initialize":
        if mode == "hang_init":
            continue
        result(request, {"protocolVersion": request["params"]["protocolVersion"],
                         "capabilities": {"tools": {"listChanged": True}, "logging": {}},
                         "serverInfo": {"name": "isolated-fixture", "version": "1"}})
    elif method == "notifications/initialized":
        if mode == "private":
            send({"jsonrpc": "2.0", "method": "notifications/message", "params": {
                "level": "error", "logger": "private-fixture", "data": os.environ["FIXTURE_LOG_SECRET"]}})
    elif method == "tools/list":
        result(request, {"tools": tools()})
    elif method == "tools/call":
        if mode == "reconnect" and launch == 1:
            os._exit(3)
        if mode == "hang_call":
            Path(config["call_file"]).write_text("received")
            time.sleep(3600)
        payload = {"arguments": request["params"].get("arguments", {}),
                   "home": os.environ.get("HERMES_HOME"), "pid": os.getpid()}
        if config.get("inspect_env"):
            payload["environment"] = {key: os.environ.get(key) for key in config["inspect_env"]}
        if "argv_secret" in config:
            payload["argv_secret"] = config["argv_secret"]
        result(request, {"content": [{"type": "text", "text": json.dumps(payload)}],
                         "structuredContent": payload, "isError": False})
        if mode == "refresh" and phase == "initial":
            phase = "refreshed"
            send({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"})
    elif method == "ping":
        result(request, {})
    elif "id" in request:
        send({"jsonrpc": "2.0", "id": request["id"],
              "error": {"code": -32601, "message": "Method not found"}})
'''


def _stdio_config(home: Path, **fixture_options) -> dict:
    env = {"HERMES_HOME": str(home)}
    for option, variable in (("stderr_secret", "FIXTURE_STDERR_SECRET"), ("log_secret", "FIXTURE_LOG_SECRET")):
        if option in fixture_options:
            env[variable] = fixture_options.pop(option)
    return {
        "command": sys.executable,
        "args": ["-u", "-c", _STDIO_SERVER, json.dumps(fixture_options)],
        "env": env,
        "connect_timeout": 5,
        "timeout": 5,
    }


@contextmanager
def _profile(home: Path, secrets: dict[str, str] | None = None):
    home.mkdir(parents=True, exist_ok=True)
    token = set_hermes_home_override(home)
    secret_token = set_secret_scope(secrets, profile_home=str(home)) if secrets is not None else None
    try:
        yield
    finally:
        if secret_token is not None:
            reset_secret_scope(secret_token)
        reset_hermes_home_override(token)


async def _until(predicate, *, timeout: float = 15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        await asyncio.sleep(0.02)
    raise AssertionError("Local MCP fixture did not reach the expected state")


def _pid_alive(pid: int) -> bool:
    return psutil.pid_exists(pid)


def _pids(path: Path) -> list[int]:
    return [int(value) for value in path.read_text().splitlines()] if path.exists() else []


class _HTTPFixture:
    def __init__(self, *, redirect: str | None = None, drop_call: bool = False,
                 registration_secret: str | None = None, oauth_challenge: str | None = None,
                 oauth_challenge_status: int = 401):
        self.requests: list[dict] = []
        self.all_requests: list[dict] = []
        self.redirect = redirect
        self.drop_call = drop_call
        self.registration_secret = registration_secret
        self.registration_requests: list[dict] = []
        self.registration_requested = threading.Event()
        self.release_registration = threading.Event()
        self.oauth_challenge = oauth_challenge
        self.oauth_challenge_status = oauth_challenge_status
        self.oauth_flow = None
        self.authorization_requests: list[dict] = []
        self.callback_requests: list[dict] = []
        self.token_requests: list[dict] = []
        self.oauth_errors: list[str] = []
        self.side_effects: list[dict] = []
        self.challenges: list[dict] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):
                pass

            def reply(self, status: int, body=None, **headers):
                encoded = json.dumps(body).encode() if body is not None else b""
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.send_header("Connection", "close")
                for name, value in headers.items():
                    self.send_header(name, value)
                self.end_headers()
                if encoded:
                    self.wfile.write(encoded)
                self.close_connection = True

            def do_GET(self):
                owner.all_requests.append({"method": "GET", "path": self.path})
                if owner.oauth_challenge is not None:
                    origin = owner.url.removesuffix("/mcp")
                    parsed = urlsplit(self.path)
                    query = {key: values[0] for key, values in parse_qs(parsed.query).items()}
                    if parsed.path == "/authorize":
                        owner.authorization_requests.append(query)
                        if (query.get("response_type") != "code" or query.get("code_challenge_method") != "S256"
                                or not query.get("state") or not query.get("code_challenge")
                                or query.get("redirect_uri") != f"{origin}/callback"):
                            owner.oauth_errors.append("invalid authorization request")
                            self.reply(400)
                            return
                        callback_query = urlencode({"code": "local-authorization-code", "state": query["state"],
                                                    "iss": origin})
                        self.reply(302, **{"Location": f'{query["redirect_uri"]}?{callback_query}'})
                        return
                    if parsed.path == "/callback":
                        owner.callback_requests.append(query)
                        try:
                            owner.oauth_flow.deliver_callback(
                                code=query.get("code"), state=query.get("state"), error=query.get("error"),
                                iss=query.get("iss"))
                        except ValueError as exc:
                            owner.oauth_errors.append(str(exc))
                            self.reply(400)
                            return
                        self.reply(200, {"callback": "delivered"})
                        return
                if owner.registration_secret is not None or owner.oauth_challenge is not None:
                    origin = owner.url.removesuffix("/mcp")
                    if "oauth-protected-resource" in self.path:
                        self.reply(200, {"resource": owner.url, "authorization_servers": [origin]})
                        return
                    if "oauth-authorization-server" in self.path or "openid-configuration" in self.path:
                        self.reply(200, {
                            "issuer": origin, "authorization_endpoint": f"{origin}/authorize",
                            "token_endpoint": f"{origin}/token", "registration_endpoint": f"{origin}/register",
                            "response_types_supported": ["code"], "code_challenge_methods_supported": ["S256"],
                        })
                        return
                self.reply(405)

            def do_DELETE(self):
                owner.all_requests.append({"method": "DELETE", "path": self.path})
                self.reply(204)

            def do_POST(self):
                request_record = {"method": "POST", "path": self.path}
                owner.all_requests.append(request_record)
                raw_body = self.rfile.read(int(self.headers["Content-Length"]))
                if self.path == "/token" and owner.oauth_challenge is not None:
                    form = {key: values[0] for key, values in parse_qs(raw_body.decode()).items()}
                    owner.token_requests.append(form)
                    if form.get("grant_type") == "authorization_code":
                        expected = owner.authorization_requests[-1] if owner.authorization_requests else {}
                        verifier = form.get("code_verifier", "")
                        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
                        valid = (form.get("code") == "local-authorization-code"
                                 and form.get("redirect_uri") == expected.get("redirect_uri")
                                 and challenge == expected.get("code_challenge"))
                    else:
                        valid = (form.get("grant_type") == "refresh_token"
                                 and form.get("refresh_token") == "refresh-token")
                    if not valid or form.get("client_id") != "isolated-local-client":
                        owner.oauth_errors.append("invalid code, PKCE verifier, client, or refresh token")
                        self.reply(400, {"error": "invalid_grant"})
                        return
                    self.reply(200, {"access_token": "new-token", "token_type": "Bearer", "expires_in": 3600,
                                     "refresh_token": "new-refresh-token"})
                    return
                body = json.loads(raw_body)
                request_record["rpc_method"] = body.get("method")
                owner.requests.append({"body": body, "headers": dict(self.headers)})
                if owner.registration_secret is not None:
                    if self.path == "/register":
                        owner.registration_requests.append(body)
                        owner.registration_requested.set()
                        if not owner.release_registration.wait(timeout=10):
                            self.reply(503)
                            return
                        self.reply(400, {"client_secret": owner.registration_secret})
                    else:
                        origin = owner.url.removesuffix("/mcp")
                        self.reply(401, {}, **{
                            "WWW-Authenticate": f'Bearer resource_metadata="{origin}/.well-known/oauth-protected-resource"'})
                    return
                if owner.redirect:
                    self.reply(307, **{"Location": owner.redirect})
                    return
                method = body["method"]
                if owner.oauth_challenge is not None:
                    if method == "tools/call":
                        owner.side_effects.append(body["params"].get("arguments", {}))
                    if method == owner.oauth_challenge and self.headers.get("Authorization") != "Bearer new-token":
                        owner.challenges.append(body)
                        origin = owner.url.removesuffix("/mcp")
                        challenge = f'Bearer resource_metadata="{origin}/.well-known/oauth-protected-resource"'
                        if owner.oauth_challenge_status == 403:
                            challenge += ', error="insufficient_scope", scope="write"'
                        self.reply(owner.oauth_challenge_status, {}, **{"WWW-Authenticate": challenge})
                        return
                if "id" not in body:
                    self.reply(202)
                    return
                if method == "initialize":
                    payload = {
                        "protocolVersion": body["params"]["protocolVersion"],
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "http-fixture", "version": "1"},
                    }
                elif method == "tools/list":
                    payload = {"tools": [{"name": "echo", "inputSchema": {"type": "object"}}]}
                elif method == "tools/call":
                    if owner.drop_call:
                        self.close_connection = True
                        self.connection.shutdown(socket.SHUT_RDWR)
                        self.connection.close()
                        return
                    payload = {"content": [{"type": "text", "text": json.dumps({
                        "authorization": self.headers.get("Authorization"),
                        "home": self.headers.get("X-Profile-Home"),
                        "arguments": body["params"].get("arguments", {}),
                    })}], "isError": False}
                else:
                    payload = {}
                self.reply(200, {"jsonrpc": "2.0", "id": body["id"], "result": payload})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_port}/mcp"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.release_registration.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def _http_config(server: _HTTPFixture, home: Path, token: str) -> dict:
    return {
        "transport": "http",
        "url": server.url,
        "headers": {"Authorization": f"Bearer {token}", "X-Profile-Home": str(home)},
        "connect_timeout": 5,
        "timeout": 5,
    }


def _payload(result: dict) -> dict:
    assert result["isError"] is False
    return json.loads(result["content"][0]["text"])


def _oauth_config(server: _HTTPFixture, home: Path) -> dict:
    return {
        "transport": "http",
        "url": server.url,
        "auth": "oauth",
        "headers": {"X-Profile-Home": str(home)},
        "connect_timeout": 5,
        "timeout": 5,
    }


async def _seed_oauth_cache(name: str, home: Path, server: _HTTPFixture, token: str, *,
                           refresh_token: str | None = None, redirect_uri: str = "https://example.invalid/oauth/callback"):
    from mcp.shared.auth import OAuthClientInformationFull, OAuthMetadata, OAuthToken
    from tools.mcp_oauth import HermesTokenStorage

    home.mkdir(parents=True, exist_ok=True)
    storage = HermesTokenStorage(name, hermes_home=home)
    base_url = server.url.removesuffix("/mcp")
    if refresh_token is not None:
        storage.bind_issuer(base_url)
    await storage.set_tokens(OAuthToken(access_token=token, expires_in=3600, refresh_token=refresh_token))
    await storage.set_client_info(OAuthClientInformationFull(
        client_id="isolated-local-client",
        redirect_uris=[redirect_uri],
        token_endpoint_auth_method="none",
    ))
    storage.save_oauth_metadata(OAuthMetadata(
        issuer=base_url, authorization_endpoint=f"{base_url}/authorize", token_endpoint=f"{base_url}/token"))
    return storage


def _spy_global_oauth_and_cache_reads(monkeypatch):
    from tools import mcp_oauth, mcp_oauth_manager

    manager_lookups = []
    cache_reads = []
    real_get_manager = mcp_oauth_manager.get_manager
    real_read_json = mcp_oauth._read_json

    def get_manager():
        manager_lookups.append(True)
        return real_get_manager()

    def read_json(path, **kwargs):
        cache_reads.append(Path(path))
        return real_read_json(path, **kwargs)

    monkeypatch.setattr(mcp_oauth_manager, "get_manager", get_manager)
    monkeypatch.setattr(mcp_oauth, "_read_json", read_json)
    return manager_lookups, cache_reads


def _deny_profile_config(reads: list):
    def deny(*_args, **_kwargs):
        reads.append(True)
        raise AssertionError("An isolated client must use its explicit config")

    return deny


def test_stdio_round_trip_keeps_profiles_and_discovery_copies_separate(tmp_path, monkeypatch):
    from hermes_cli import env_loader

    source_var = "PERSONAL_SOURCE_TOKEN"
    explicit_var = "PERSONAL_EXPLICIT_TOKEN"
    ambient_var = "PERSONAL_AMBIENT_TOKEN"
    monkeypatch.setitem(env_loader._SECRET_SOURCES, source_var, "command")
    monkeypatch.setenv(source_var, "launch-source-secret")
    monkeypatch.setenv(ambient_var, "launch-ambient-secret")

    async def scenario():
        clients = [IsolatedMCPClient("same-name"), IsolatedMCPClient("same-name")]
        homes = [tmp_path / "a", tmp_path / "b"]
        try:
            for client, home in zip(clients, homes):
                with _profile(home, {source_var: f"{home.name}-source-secret"}):
                    config = _stdio_config(home, inspect_env=[source_var, explicit_var, ambient_var])
                    config["env"][explicit_var] = f"{home.name}-explicit-secret"
                    await client.start(config)
            discovered = clients[0].discovered_tools()
            assert discovered[0]["inputSchema"]["properties"]["nonce"]["type"] == "string"
            assert "input_schema" not in discovered[0]
            assert all(value is not None for value in discovered[0].values())
            discovered[0]["inputSchema"]["properties"]["nonce"]["type"] = "integer"
            discovered.clear()
            assert clients[0].discovered_tools()[0]["inputSchema"]["properties"]["nonce"] == {"type": "string"}
            for index, nonce in ((0, "a-first"), (1, "b"), (0, "a-again")):
                with _profile(homes[index]):
                    result = await clients[index].call_tool("echo", {"nonce": nonce})
                payload = _payload(result)
                assert payload["home"] == str(homes[index])
                assert payload["arguments"] == {"nonce": nonce}
                assert payload["environment"] == {
                    source_var: None, explicit_var: f"{homes[index].name}-explicit-secret", ambient_var: None}
                assert result["structuredContent"] == payload
                assert "structured_content" not in result
        finally:
            for client in clients:
                await client.close()

    asyncio.run(scenario())


def test_http_a_b_a_uses_each_clients_explicit_headers_and_profile(tmp_path, monkeypatch):
    from tools import mcp_tool_config

    config_reads = []
    monkeypatch.setattr(mcp_tool_config, "_load_mcp_config", _deny_profile_config(config_reads))

    async def scenario(server):
        clients = [IsolatedMCPClient("same-http"), IsolatedMCPClient("same-http")]
        homes = [tmp_path / "a", tmp_path / "b"]
        tokens = ["profile-a-token", "profile-b-token"]
        try:
            for client, home, token in zip(clients, homes, tokens):
                with _profile(home):
                    (home / "config.yaml").write_text("mcp_servers: invalid-profile-config\n")
                    config = _http_config(server, home, token)
                    config["sampling"] = {"enabled": True}
                    config["elicitation"] = {"enabled": True}
                    await client.start(config)
            for index in (0, 1, 0):
                with _profile(homes[index]):
                    payload = _payload(await clients[index].call_tool("echo", {"index": index}))
                assert payload == {"authorization": f"Bearer {tokens[index]}",
                                   "home": str(homes[index]), "arguments": {"index": index}}
        finally:
            for client in clients:
                await client.close()

    with _HTTPFixture() as server:
        asyncio.run(scenario(server))
        calls = [request for request in server.requests if request["body"]["method"] == "tools/call"]
        assert [request["headers"]["Authorization"] for request in calls] == [
            "Bearer profile-a-token", "Bearer profile-b-token", "Bearer profile-a-token"]
        handshakes = [request["body"] for request in server.requests
                      if request["body"]["method"] == "initialize"]
        assert len(handshakes) == 2
        for handshake in handshakes:
            assert "sampling" not in handshake["params"]["capabilities"]
            assert "elicitation" not in handshake["params"]["capabilities"]
    assert config_reads == []


def test_oauth_requires_explicit_home_before_ambient_cache_or_network(tmp_path, monkeypatch):
    from tools import mcp_oauth_manager

    name = "isolated-oauth-implicit"
    ambient_home = tmp_path / "ambient"
    manager = mcp_oauth_manager.get_manager()

    async def scenario(server):
        with _profile(ambient_home):
            storage = await _seed_oauth_cache(name, ambient_home, server, "ambient-personal-token")
            ambient_provider = manager.get_or_build_provider(name, server.url, None)
            assert ambient_provider is not None
            snapshot = storage.snapshot()
            manager_lookups, cache_reads = _spy_global_oauth_and_cache_reads(monkeypatch)
            client = IsolatedMCPClient(name)
            try:
                with pytest.raises(ValueError, match="oauth_home"):
                    await client.start(_oauth_config(server, ambient_home))
                assert manager_lookups == []
                assert cache_reads == []
                assert server.all_requests == []
                assert storage.snapshot() == snapshot
                assert manager.get_or_build_provider(name, server.url, None) is ambient_provider
            finally:
                await client.close()
                manager.evict(name, hermes_home=ambient_home)

    with _HTTPFixture() as server:
        asyncio.run(scenario(server))


def test_explicit_oauth_homes_keep_a_b_a_private_under_ambient_profile_switches(tmp_path, monkeypatch, caplog):
    from tools import mcp_oauth_manager

    name = "isolated-private-oauth"
    ambient_homes = [tmp_path / "ambient-a", tmp_path / "ambient-b"]
    private_homes = [tmp_path / "private-a", tmp_path / "private-b"]
    private_tokens = ["private-oauth-a-token", "private-oauth-b-token"]
    manager = mcp_oauth_manager.get_manager()

    async def scenario(server):
        ambient_stores = []
        ambient_providers = []
        for index, home in enumerate(ambient_homes):
            with _profile(home):
                ambient_stores.append(await _seed_oauth_cache(name, home, server, f"ambient-{index}-token"))
                ambient_providers.append(manager.get_or_build_provider(name, server.url, None))
        for home, token in zip(private_homes, private_tokens):
            await _seed_oauth_cache(name, home, server, token)
        snapshots = [storage.snapshot() for storage in ambient_stores]
        manager_lookups, cache_reads = _spy_global_oauth_and_cache_reads(monkeypatch)
        clients = []
        try:
            with caplog.at_level(logging.DEBUG):
                for index, home in enumerate(private_homes):
                    with _profile(ambient_homes[index]):
                        client = IsolatedMCPClient(name, oauth_home=home)
                        clients.append(client)
                        await client.start(_oauth_config(server, home))
                for index in (0, 1, 0):
                    with _profile(ambient_homes[1 - index]):
                        result = await clients[index].call_tool("echo", {"index": index})
                    assert _payload(result) == {
                        "authorization": f"Bearer {private_tokens[index]}",
                        "home": str(private_homes[index]), "arguments": {"index": index}}
                for index, home in enumerate(ambient_homes):
                    with _profile(home):
                        assert manager.get_or_build_provider(name, server.url, None) is ambient_providers[index]
        finally:
            for client in clients:
                await client.close()
            for home in ambient_homes:
                manager.evict(name, hermes_home=home)
        assert manager_lookups == []
        assert cache_reads, "The real OAuth provider must read its explicit token storage"
        assert all(any(path.is_relative_to(home) for home in private_homes) for path in cache_reads)
        assert [storage.snapshot() for storage in ambient_stores] == snapshots
        assert all(provider is not None for provider in ambient_providers)

    with _HTTPFixture() as server:
        asyncio.run(scenario(server))
        calls = [request for request in server.requests if request["body"]["method"] == "tools/call"]
        assert [request["headers"]["Authorization"] for request in calls] == [
            f"Bearer {private_tokens[0]}", f"Bearer {private_tokens[1]}", f"Bearer {private_tokens[0]}"]
    for private_value in (name, *map(str, private_homes), *private_tokens):
        assert private_value not in caplog.text


def test_private_oauth_registration_failure_hides_secret_without_muting_concurrent_sdk_logs(tmp_path, caplog):
    from mcp.shared.auth import OAuthToken
    from tools.mcp_oauth import HermesTokenStorage

    registration_secret = "private-oauth-registration-response-0cc6cf"
    normal_canary = "normal-concurrent-sdk-oauth-warning-e82f6d"
    name = "isolated-oauth-registration-error"
    private_home = tmp_path / "private"

    async def scenario(server):
        await HermesTokenStorage(name, hermes_home=private_home).set_tokens(
            OAuthToken(access_token="synthetic-cached-oauth-token", expires_in=3600))
        client = IsolatedMCPClient(name, oauth_home=private_home)

        async def emit_normal_diagnostic():
            requested = await asyncio.to_thread(server.registration_requested.wait, 5)
            assert requested, "The real SDK must reach dynamic client registration"
            logging.getLogger("mcp.client.auth.oauth2").warning(normal_canary)
            server.release_registration.set()

        ordinary_task = asyncio.create_task(emit_normal_diagnostic())
        config = _oauth_config(server, private_home)
        config["oauth"] = {"cimd": False, "redirect_port": 43210}
        config["connect_timeout"] = 3
        try:
            with pytest.raises(Exception):
                await asyncio.wait_for(client.start(config), timeout=8)
            await ordinary_task
            assert server.registration_requests
            assert all(request.get("redirect_uris") for request in server.registration_requests)
        finally:
            server.release_registration.set()
            if not ordinary_task.done():
                ordinary_task.cancel()
            await asyncio.gather(ordinary_task, return_exceptions=True)
            await client.close()

    with _HTTPFixture(registration_secret=registration_secret) as server:
        with caplog.at_level(logging.DEBUG):
            asyncio.run(scenario(server))
    assert registration_secret not in caplog.text
    assert normal_canary in caplog.text


async def _complete_local_oauth_browser_flow(flow):
    import httpx2

    authorization_url = await flow.wait_for_authorization_url(timeout=5)
    assert urlsplit(authorization_url).hostname == "127.0.0.1"
    async with httpx2.AsyncClient(follow_redirects=True, trust_env=False) as browser:
        response = await browser.get(authorization_url)
    assert response.status_code == 200
    assert response.json() == {"callback": "delivered"}


@pytest.mark.parametrize("status", [401, 403])
def test_isolated_oauth_challenge_does_not_replay_a_tool_call_or_begin_authorization(tmp_path, status):
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow, dashboard_oauth_flow

    async def scenario(server):
        name = "isolated-oauth-no-replay"
        callback_url = server.url.removesuffix("/mcp") + "/callback"
        await _seed_oauth_cache(name, tmp_path, server, "old-token", redirect_uri=callback_url)
        flow = DashboardOAuthFlow(flow_id="local-isolated", server_name=name, profile=None,
                                  hermes_home=str(tmp_path), redirect_uri=callback_url)
        server.oauth_flow = flow
        client = IsolatedMCPClient(name, oauth_home=tmp_path)
        try:
            with dashboard_oauth_flow(flow):
                await client.start(_oauth_config(server, tmp_path))
                with pytest.raises(Exception):
                    await client.call_tool("echo", {"nonce": "one-side-effect"}, timeout=3)
            assert server.side_effects == [{"nonce": "one-side-effect"}]
            assert len(server.challenges) == 1
            assert flow.authorization_url is None
            assert server.authorization_requests == []
            assert server.callback_requests == []
            assert server.token_requests == []
            assert server.oauth_errors == []
        finally:
            flow.mark_error("test cleanup", cancelled=True)
            await client.close()

    with _HTTPFixture(oauth_challenge="tools/call", oauth_challenge_status=status) as server:
        asyncio.run(scenario(server))


def test_isolated_oauth_refreshes_an_expired_token_before_sending_one_tool_call(tmp_path):
    from mcp.shared.auth import OAuthToken

    async def scenario(server):
        name = "isolated-oauth-preflight-refresh"
        storage = await _seed_oauth_cache(name, tmp_path, server, "old-token", refresh_token="refresh-token")
        client = IsolatedMCPClient(name, oauth_home=tmp_path)
        try:
            await client.start(_oauth_config(server, tmp_path))
            await storage.set_tokens(OAuthToken(
                access_token="expired-token", expires_in=0, refresh_token="refresh-token"))
            result = await client.call_tool("echo", {"nonce": "refresh-before-dispatch"}, timeout=5)
            assert _payload(result)["authorization"] == "Bearer new-token"
            assert server.side_effects == [{"nonce": "refresh-before-dispatch"}]
            assert [form["grant_type"] for form in server.token_requests] == ["refresh_token"]
            assert server.challenges == []
            assert server.authorization_requests == []
            assert server.callback_requests == []
            methods = ["token" if request["path"] == "/token" else request.get("rpc_method")
                       for request in server.all_requests]
            assert methods.index("token") < methods.index("tools/call")
            assert server.oauth_errors == []
        finally:
            await client.close()

    with _HTTPFixture(oauth_challenge="tools/call") as server:
        asyncio.run(scenario(server))


def test_native_oauth_can_reauthorize_and_retry_tool_discovery(tmp_path):
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow, dashboard_oauth_flow
    from tools.mcp_oauth_manager import get_manager
    from tools.mcp_tool import MCPServerTask

    async def scenario(server):
        name = "native-oauth-discovery-retry"
        callback_url = server.url.removesuffix("/mcp") + "/callback"
        await _seed_oauth_cache(name, tmp_path, server, "old-token", redirect_uri=callback_url)
        flow = DashboardOAuthFlow(flow_id="local-native", server_name=name, profile=None,
                                  hermes_home=str(tmp_path), redirect_uri=callback_url)
        server.oauth_flow = flow
        native = MCPServerTask(name)
        browser = None
        try:
            with _profile(tmp_path), dashboard_oauth_flow(flow):
                browser = asyncio.create_task(_complete_local_oauth_browser_flow(flow))
                config = _oauth_config(server, tmp_path)
                config["skip_preflight"] = True
                await asyncio.wait_for(native.start(config), timeout=8)
                await browser
            listings = [request for request in server.requests if request["body"].get("method") == "tools/list"]
            assert [request["headers"]["Authorization"] for request in listings] == [
                "Bearer old-token", "Bearer new-token"]
            assert len(server.challenges) == 1
            assert len(server.authorization_requests) == 1
            assert len(server.callback_requests) == 1
            assert [form["grant_type"] for form in server.token_requests] == ["authorization_code"]
            assert server.oauth_errors == []
        finally:
            flow.mark_error("test cleanup", cancelled=True)
            if browser is not None and not browser.done():
                browser.cancel()
                await asyncio.gather(browser, return_exceptions=True)
            await native.shutdown()
            get_manager().evict(name, hermes_home=tmp_path)

    with _HTTPFixture(oauth_challenge="tools/list") as server:
        asyncio.run(scenario(server))


def test_stdio_list_changed_updates_only_the_isolated_discovery(tmp_path):
    from tools.registry import registry

    before = set(registry.get_all_tool_names())

    async def scenario():
        client = IsolatedMCPClient("isolated-dynamic")
        try:
            await client.start(_stdio_config(tmp_path, mode="refresh"))
            assert [tool["name"] for tool in client.discovered_tools()] == ["echo"]
            await client.call_tool("echo", {})
            await _until(lambda: [tool["name"] for tool in client.discovered_tools()] == ["after_refresh"])
            assert _payload(await client.call_tool("after_refresh", {"nonce": "updated"}))["arguments"] == {
                "nonce": "updated"}
            assert set(registry.get_all_tool_names()) == before
        finally:
            await client.close()
        assert set(registry.get_all_tool_names()) == before

    asyncio.run(scenario())


def test_stdio_stderr_and_log_notifications_never_enter_logs(tmp_path, monkeypatch, caplog):
    from hermes_cli import process_identity

    stderr_secret = "private-stdio-stderr-5fd2ed"
    log_secret = "private-mcp-notification-bd3bc1"
    argv_secret = "private-stdio-argument-8d4dfc"
    ledger_file = tmp_path / "spawn-ledger.json"
    monkeypatch.setattr(process_identity, "_ledger_path", lambda: ledger_file)
    logfile = tmp_path / "captured.log"
    handler = logging.FileHandler(logfile)
    handler.setLevel(logging.DEBUG)
    root = logging.getLogger()
    root.addHandler(handler)

    async def scenario(stderr=None):
        client = IsolatedMCPClient("isolated-private", stderr=stderr)
        try:
            await client.start(_stdio_config(tmp_path, mode="private", stderr_secret=stderr_secret,
                                             log_secret=log_secret, argv_secret=argv_secret))
            payload = _payload(await client.call_tool("echo", {"nonce": "barrier"}))
            assert payload["arguments"] == {"nonce": "barrier"}
            assert payload["argv_secret"] == argv_secret
            child = psutil.Process(payload["pid"])
            assert argv_secret in " ".join(child.cmdline())
            entries = json.loads(ledger_file.read_text())
            assert all(argv_secret not in entry["argv"] for entry in entries)
            entry = next(entry for entry in entries if entry["pid"] == child.pid)
            assert entry["create_time"] == pytest.approx(child.create_time(), rel=0, abs=0.01)
            assert entry["purpose"] == "mcp-helper"
            assert entry["spawner_pid"] == psutil.Process().pid
            assert entry["argv"] == ""
        finally:
            await client.close()

    try:
        with caplog.at_level(logging.DEBUG):
            asyncio.run(scenario())
    finally:
        root.removeHandler(handler)
        handler.close()
    for secret in (stderr_secret, log_secret, argv_secret):
        assert secret not in caplog.text
        for path in tmp_path.rglob("*"):
            if path.is_file():
                assert secret not in path.read_text(errors="replace"), path

    with (tmp_path / "caller-owned-stderr.log").open("w+", encoding="utf-8") as stderr:
        asyncio.run(scenario(stderr))
        assert not stderr.closed
        stderr.seek(0)
        captured = stderr.read()
        assert stderr_secret in captured
        assert log_secret not in captured
    assert log_secret not in caplog.text
    assert argv_secret not in caplog.text


def test_http_redirect_is_not_followed(tmp_path):
    async def scenario(source):
        client = IsolatedMCPClient("isolated-redirect")
        try:
            with pytest.raises(Exception):
                await asyncio.wait_for(client.start(_http_config(source, tmp_path, "redirect-token")), 10)
        finally:
            await client.close()

    with _HTTPFixture() as target, _HTTPFixture(redirect=target.url) as source:
        asyncio.run(scenario(source))
        assert source.requests, "The real SDK must reach the redirecting endpoint"
        assert target.requests == []


def test_http_lost_response_does_not_repeat_a_tool_call(tmp_path):
    async def scenario(server):
        client = IsolatedMCPClient("isolated-no-retry")
        try:
            await client.start(_http_config(server, tmp_path, "one-call-token"))
            with pytest.raises(Exception):
                await asyncio.wait_for(client.call_tool("echo", {"nonce": "side-effect"}, timeout=3), 8)
        finally:
            await client.close()

    with _HTTPFixture(drop_call=True) as server:
        asyncio.run(scenario(server))
        calls = [request for request in server.requests if request["body"]["method"] == "tools/call"]
        assert len(calls) == 1
        assert calls[0]["body"]["params"]["arguments"] == {"nonce": "side-effect"}


def test_stdio_reconnect_retains_explicit_config_without_profile_reads(tmp_path, monkeypatch):
    from tools import mcp_tool_config

    config_reads = []
    monkeypatch.setattr(mcp_tool_config, "_load_mcp_config", _deny_profile_config(config_reads))
    pid_file = tmp_path / "launches.txt"

    async def scenario():
        client = IsolatedMCPClient("isolated-reconnect")
        try:
            config = _stdio_config(tmp_path, mode="reconnect", pid_file=str(pid_file))
            config["keepalive_interval"] = 5
            await client.start(config)
            with pytest.raises(Exception):
                await asyncio.wait_for(client.call_tool("echo", {"nonce": "crash"}, timeout=3), 8)
            await _until(lambda: len(_pids(pid_file)) >= 2, timeout=20)
            await _until(lambda: [tool["name"] for tool in client.discovered_tools()] == ["after_reconnect"])
            result = await client.call_tool("after_reconnect", {"nonce": "after-reconnect"}, timeout=5)
            assert _payload(result)["arguments"] == {"nonce": "after-reconnect"}
            assert _payload(result)["home"] == str(tmp_path)
            assert _payload(result)["pid"] != _pids(pid_file)[0]
        finally:
            await client.close()
        assert all(not _pid_alive(pid) for pid in _pids(pid_file))

    asyncio.run(scenario())
    assert config_reads == []


@pytest.mark.parametrize("action", ["cancel_start", "cancel_call", "close"])
def test_stdio_cancellation_and_close_reap_the_real_child(tmp_path, action):
    pid_file = tmp_path / "pid.txt"
    call_file = tmp_path / "call.txt"
    mode = {"cancel_start": "hang_init", "cancel_call": "hang_call", "close": "normal"}[action]

    async def scenario():
        client = IsolatedMCPClient(f"isolated-{action}")
        pending = None
        try:
            pending = asyncio.create_task(client.start(_stdio_config(
                tmp_path, mode=mode, pid_file=str(pid_file), call_file=str(call_file))))
            await _until(lambda: _pids(pid_file))
            pid = _pids(pid_file)[0]
            assert _pid_alive(pid)
            if action == "cancel_start":
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
                await _until(lambda: not _pid_alive(pid))
            else:
                await pending
                if action == "cancel_call":
                    pending = asyncio.create_task(client.call_tool("echo", {}))
                    await _until(call_file.exists)
                    pending.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await pending
                await client.close()
                await _until(lambda: not _pid_alive(pid))
        finally:
            if pending is not None and not pending.done():
                pending.cancel()
                try:
                    await pending
                except asyncio.CancelledError:
                    pass
            await client.close()
        assert _pids(pid_file)
        assert all(not _pid_alive(pid) for pid in _pids(pid_file))

    asyncio.run(scenario())
