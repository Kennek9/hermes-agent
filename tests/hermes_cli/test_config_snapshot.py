"""Real cold-process config snapshots must preserve profile bytes and metadata."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[2]

PROBE = """
import hashlib
import json
import logging
import os
from pathlib import Path
import stat
import sys

root = Path(os.environ['SNAPSHOT_ROOT'])
home = Path(os.environ['HERMES_HOME'])
log_path = root / 'observed.log'
handler = logging.FileHandler(log_path)
logging.getLogger().addHandler(handler)
active = False
write_attempts = []

def audit(event, args):
    if not active:
        return
    if event in {'os.mkdir', 'os.chmod', 'os.chown', 'os.remove', 'os.rename', 'os.rmdir', 'os.utime', 'os.link', 'os.symlink', 'os.truncate'}:
        write_attempts.append(event)
    elif event == 'open' and isinstance(args[2], int) and args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
        write_attempts.append(event)

sys.addaudithook(audit)

def fingerprint():
    result = {}
    for path in [root, *sorted(root.rglob('*'))]:
        info = path.lstat()
        result[str(path.relative_to(root))] = (
            info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
        )
    return result

def inspect_read(read):
    global active
    handler.flush()
    before = fingerprint()
    write_attempts.clear()
    active = True
    try:
        value = read()
    finally:
        active = False
        handler.flush()
        assert not write_attempts, write_attempts
        assert fingerprint() == before, 'read changed profile content, directory entries or metadata'
    return value

def context(snapshot):
    return PluginContext(PluginManifest(name='probe', source='user'), PluginManager(scope_key=str(home)), config_snapshot=snapshot)

# The target profile is selected before the FIRST native import. Measure this
# path too: config's provider metadata discovery must not initialize the home.
def import_native():
    global cfg, FailedConfigRead, PluginContext, PluginManager, PluginManifest
    from hermes_cli import config as cfg
    from hermes_cli.config_read_errors import FailedConfigRead
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

inspect_read(import_native)

"""


def run_probe(tmp_path: Path, action: str) -> dict:
    managed = tmp_path / "managed"
    managed.mkdir(exist_ok=True)
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": str(ROOT),
        "PYTHONDONTWRITEBYTECODE": "1",
        "HERMES_HOME": str(tmp_path / "home"),
        "HERMES_MANAGED_DIR": str(managed),
        "SNAPSHOT_ROOT": str(tmp_path),
        "SNAPSHOT_LABEL": "expanded-user",
        "SNAPSHOT_MANAGED_BASE": "https://managed.example/v1",
    })
    env.pop("SNAPSHOT_ABSENT", None)
    process = subprocess.run(
        [sys.executable, "-B", "-c", PROBE + textwrap.dedent(action)],
        cwd=tmp_path, env=env, text=True, capture_output=True, timeout=60,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    assert not process.stderr, process.stderr
    return json.loads(process.stdout)


def test_snapshot_context_is_pure_and_uses_the_canonical_pipeline(tmp_path):
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    (home / "config.yaml").write_text("""max_turns: 23
agent:
  max_turns: null
model:
  name: user/model
  provider: 2
plugins:
  entries:
    probe:
      settings:
        label: ${env:SNAPSHOT_LABEL}
        missing: ${env:SNAPSHOT_ABSENT}
        external: ${vault:unresolved}
        zero: 0
        nil: null
      config:
        legacy_only: compatible
        label: legacy-shadowed
        nil: legacy-shadowed
""", encoding="utf-8")
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text("""model:
  api_base: ${env:SNAPSHOT_MANAGED_BASE}
plugins:
  entries:
    probe:
      settings:
        managed: pinned
""", encoding="utf-8")
    result = run_probe(tmp_path, """
        def read():
            snapshot = cfg.load_config_snapshot()
            assert snapshot['agent']['max_turns'] == 23
            assert snapshot['model']['default'] == 'user/model'
            assert snapshot['model']['provider'] == '2'
            assert snapshot['model']['base_url'] == 'https://managed.example/v1'
            assert snapshot['terminal']['backend'] == cfg.DEFAULT_CONFIG['terminal']['backend']
            ctx = context(snapshot)
            assert ctx.get_config('label') == 'expanded-user'
            assert ctx.get_config('legacy_only') == 'compatible'
            assert ctx.get_config('managed') == 'pinned'
            assert ctx.get_config('missing') == '${env:SNAPSHOT_ABSENT}'
            assert ctx.get_config('external') == '${vault:unresolved}'
            assert ctx.get_config('zero', 99) == 0
            assert ctx.get_config('nil', 99) is None
            assert ctx.get_config('absent', 'fallback') == 'fallback'
            snapshot['plugins']['entries']['probe']['settings']['label'] = 'mutated'
            assert ctx.get_config('label') == 'expanded-user'
            try:
                ctx.get_config('../escape')
            except ValueError:
                pass
            else:
                raise AssertionError('invalid plugin path accepted')
            return cfg.load_config_snapshot()
        snapshot = inspect_read(read)
        assert not (home / 'SOUL.md').exists()
        assert not (home / 'backups').exists()
        normal = cfg.load_config()
        assert normal == snapshot
        assert (home / 'SOUL.md').is_file()
        assert (home / 'logs' / 'curator').is_dir()
        assert list((home / 'backups' / 'config').glob('config.yaml.good.*'))
        ctx = PluginContext(PluginManifest(name='probe', source='user'), PluginManager(scope_key=str(home)))
        assert ctx.get_config('label') == 'expanded-user'
        assert cfg.load_config_readonly() is cfg.load_config_readonly()
        print(json.dumps({'pure': True, 'normal_initialization': True, 'normal_cache': True}))
    """)
    assert result == {"pure": True, "normal_initialization": True, "normal_cache": True}


@pytest.mark.parametrize("existing", [False, True])
def test_missing_config_never_initializes_an_empty_or_absent_home(tmp_path, existing):
    if existing:
        (tmp_path / "home").mkdir(mode=0o700)
    result = run_probe(tmp_path, """
        snapshot = inspect_read(cfg.load_config_snapshot)
        assert not isinstance(snapshot, FailedConfigRead)
        assert snapshot == cfg.DEFAULT_CONFIG
        cfg.load_config()
        assert (home / 'SOUL.md').is_file()
        assert (home / 'sessions').is_dir()
        assert stat.S_IMODE(home.stat().st_mode) == 0o700
        print(json.dumps({'pure': True, 'normal_initialization': True}))
    """)
    assert result["normal_initialization"]


@pytest.mark.parametrize("malformed", ["broken: [unterminated", "- invalid-root", "invalid-scalar"])
def test_invalid_user_config_keeps_failure_marker_without_backup_or_log(tmp_path, malformed):
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    (home / "config.yaml").write_text(malformed, encoding="utf-8")
    result = run_probe(tmp_path, """
        snapshot = inspect_read(cfg.load_config_snapshot)
        assert isinstance(snapshot, FailedConfigRead)
        assert isinstance(snapshot.read_error, Exception)
        ctx = inspect_read(lambda: context(snapshot))
        assert isinstance(ctx._config_snapshot, FailedConfigRead)
        assert isinstance(ctx._config_snapshot.read_error, type(snapshot.read_error))
        assert snapshot['agent'] == cfg.DEFAULT_CONFIG['agent']
        assert not (home / 'backups').exists()
        assert not log_path.read_text()
        print(json.dumps({'failed_read': True, 'pure': True}))
    """)
    assert result["failed_read"]


def test_real_read_error_is_a_pure_failed_read(tmp_path):
    home = tmp_path / "home"
    (home / "config.yaml").mkdir(parents=True)
    result = run_probe(tmp_path, """
        snapshot = inspect_read(cfg.load_config_snapshot)
        assert isinstance(snapshot, FailedConfigRead)
        assert isinstance(snapshot.read_error, OSError)
        assert not (home / 'backups').exists()
        print(json.dumps({'failed_read': True}))
    """)
    assert result["failed_read"]


@pytest.mark.parametrize("backup", ["good", "broken-newest"])
def test_cold_fallback_reads_only_the_newest_good_backup(tmp_path, backup):
    home = tmp_path / "home"
    copies = home / "backups" / "config"
    copies.mkdir(parents=True)
    (home / "config.yaml").write_text("broken: [unterminated", encoding="utf-8")
    (copies / "config.yaml.good.20261017-000001").write_text("approvals:\n  deny: [protected]\n", encoding="utf-8")
    if backup == "broken-newest":
        (copies / "config.yaml.good.20261017-000002").write_text("broken: [unterminated", encoding="utf-8")
    result = run_probe(tmp_path, """
        snapshot = inspect_read(cfg.load_config_snapshot)
        assert isinstance(snapshot, FailedConfigRead)
        has_broken_newest = (home / 'backups/config/config.yaml.good.20261017-000002').exists()
        assert snapshot['approvals']['deny'] == (cfg.DEFAULT_CONFIG['approvals']['deny'] if has_broken_newest else ['protected'])
        assert not log_path.read_text()
        print(json.dumps({'pure': True}))
    """)
    assert result["pure"]


def test_existing_normal_last_good_and_normal_error_diagnostics_are_preserved(tmp_path):
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    (home / "config.yaml").write_text("approvals:\n  deny: [protected]\n", encoding="utf-8")
    result = run_probe(tmp_path, """
        good = cfg.load_config()
        (home / 'config.yaml').write_text('broken: [unterminated', encoding='utf-8')
        snapshot = inspect_read(cfg.load_config_snapshot)
        assert isinstance(snapshot, FailedConfigRead) and snapshot == good
        assert not list((home / 'backups/config').glob('config.yaml.corrupt.*'))
        assert not log_path.read_text()
        # Normal mode still emits its error and creates its corrupt backup.
        import contextlib
        import io
        error = io.StringIO()
        with contextlib.redirect_stderr(error):
            normal = cfg.load_config()
        assert isinstance(normal, FailedConfigRead) and normal == good
        assert 'formatting error' in error.getvalue()
        assert log_path.read_text()
        assert list((home / 'backups/config').glob('config.yaml.corrupt.*'))
        print(json.dumps({'normal_error_behavior': True}))
    """)
    assert result["normal_error_behavior"]


def test_malformed_managed_config_is_ignored_without_logging(tmp_path):
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    (home / "config.yaml").write_text("agent:\n  max_turns: 17\n", encoding="utf-8")
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text("broken: [unterminated", encoding="utf-8")
    result = run_probe(tmp_path, """
        snapshot = inspect_read(cfg.load_config_snapshot)
        assert not isinstance(snapshot, FailedConfigRead)
        assert snapshot['agent']['max_turns'] == 17
        assert not log_path.read_text()
        cfg.load_config()
        assert 'managed scope: failed to parse' in log_path.read_text()
        print(json.dumps({'managed_fallback': True}))
    """)
    assert result["managed_fallback"]


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16", "utf-16-be", "latin-1"])
@pytest.mark.parametrize("explicit_backend", [False, True])
def test_cold_dotenv_snapshot_context_preserves_files_and_environment_rules(tmp_path, encoding, explicit_backend):
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    body = """  READONLY_ROOT=user  
READONLY_CHAIN=${READONLY_ROOT}-chain
READONLY_LATIN=café
COPILOT_ACP_BASE_URL=https://user.example
HERMES_DASHBOARD_SESSION_TOKEN=file-value
READONLY_API_KEY=secrét
TERMINAL_ENV=docker
TERMINAL_TIMEOUT=77
unparseable broken statement
"""
    if encoding == "latin-1":
        # An unchanged latin-1 file follows the normal loader's latin-1 fallback.
        body = body.replace("  READONLY_ROOT=user  ", "READONLY_ROOT=user")
    if encoding == "utf-8-sig":
        body = body.replace("READONLY_CHAIN", "READONLY_\x00CHAIN")
    data = body.encode(encoding)
    if encoding == "utf-16-be":
        data = b"\xfe\xff" + data
    (home / ".env").write_bytes(data)
    (home / ".op.env").write_text("OP_SERVICE_ACCOUNT_TOKEN=bootstrap\nREADONLY_ROOT=op-shadow\n", encoding="utf-8")
    (tmp_path / "project.env").write_text("READONLY_ROOT=project-shadow\nREADONLY_PROJECT=${READONLY_ROOT}-project\n", encoding="utf-8")
    backend = "  backend: local\n" if explicit_backend else ""
    (home / "config.yaml").write_text("terminal:\n  timeout: 600\n" + backend + """plugins:
  entries:
    probe:
      settings:
        label: ${env:READONLY_ROOT}
""", encoding="utf-8")
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / ".env").write_text("READONLY_ROOT=managed\nREADONLY_MANAGED=${READONLY_ROOT}-managed\n", encoding="utf-8")
    (managed / "config.yaml").write_text("plugins:\n  entries:\n    probe:\n      settings:\n        managed: pinned\n", encoding="utf-8")
    result = run_probe(tmp_path, """
        os.environ['HERMES_ACP_AUTH_METHOD'] = 'inherited'
        os.environ['HERMES_DASHBOARD_SESSION_TOKEN'] = 'parent-minted'
        os.environ.pop('OP_SERVICE_ACCOUNT_TOKEN', None)
        def read():
            from hermes_cli.env_loader import load_hermes_dotenv
            loaded = load_hermes_dotenv(hermes_home=home, project_env=root / 'project.env', load_external_secrets=False, read_only_files=True)
            assert loaded == [home / '.env', root / 'project.env']
            snapshot = cfg.load_config_snapshot()
            ctx = context(snapshot)
            assert ctx.get_config('label') == 'managed'
            assert ctx.get_config('managed') == 'pinned'
            assert os.environ['READONLY_ROOT'] == 'managed'
            assert os.environ['READONLY_CHAIN'] == 'user-chain'
            assert os.environ['READONLY_PROJECT'] == 'user-project'
            assert os.environ['READONLY_MANAGED'] == 'managed-managed'
            assert os.environ['READONLY_LATIN'] == 'café'
            assert os.environ['COPILOT_ACP_BASE_URL'] == 'https://user.example'
            assert 'HERMES_ACP_AUTH_METHOD' not in os.environ
            assert os.environ['HERMES_DASHBOARD_SESSION_TOKEN'] == 'parent-minted'
            assert os.environ['OP_SERVICE_ACCOUNT_TOKEN'] == 'bootstrap'
            assert os.environ['READONLY_API_KEY'] == 'secrt'
            assert os.environ['TERMINAL_TIMEOUT'] == '600'
            expected_backend = 'local' if 'backend:' in (home / 'config.yaml').read_text() else 'docker'
            assert os.environ['TERMINAL_ENV'] == expected_backend
            return {key: os.environ[key] for key in ['READONLY_ROOT', 'READONLY_CHAIN', 'READONLY_PROJECT', 'READONLY_MANAGED', 'READONLY_LATIN', 'COPILOT_ACP_BASE_URL', 'HERMES_DASHBOARD_SESSION_TOKEN', 'OP_SERVICE_ACCOUNT_TOKEN', 'READONLY_API_KEY', 'TERMINAL_ENV', 'TERMINAL_TIMEOUT']}
        first = inspect_read(read)
        assert inspect_read(read) == first
        assert not (home / 'SOUL.md').exists()
        assert not (home / 'backups').exists()
        assert not log_path.read_text()
        # Normal loading still sanitizes on disk and initializes the same home.
        import contextlib
        import io
        from hermes_cli.env_loader import load_hermes_dotenv
        with contextlib.redirect_stderr(io.StringIO()):
            load_hermes_dotenv(hermes_home=home, project_env=root / 'project.env', load_external_secrets=False)
        assert {key: os.environ[key] for key in first} == first
        assert (home / 'SOUL.md').is_file()
        assert list((home / 'backups/config').glob('config.yaml.good.*'))
        print(json.dumps({'pure': True, 'normal_parity': True}))
    """)
    assert result == {"pure": True, "normal_parity": True}


def test_readonly_dotenv_rejects_external_sources_before_any_write(tmp_path):
    result = run_probe(tmp_path, """
        def reject():
            from hermes_cli.env_loader import load_hermes_dotenv
            try:
                load_hermes_dotenv(hermes_home=home, read_only_files=True)
            except ValueError as exc:
                assert 'load_external_secrets=False' in str(exc)
            else:
                raise AssertionError('external source hydration accepted')
        inspect_read(reject)
        print(json.dumps({'rejected': True}))
    """)
    assert result["rejected"]


def test_readonly_dotenv_error_leaves_unsupported_encoding_untouched(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_bytes("KEY=value\n".encode("utf-32"))
    result = run_probe(tmp_path, """
        def reject():
            from hermes_cli.env_loader import load_hermes_dotenv
            try:
                load_hermes_dotenv(hermes_home=home, load_external_secrets=False, read_only_files=True)
            except ValueError:
                return
            raise AssertionError('unsupported UTF-32 silently accepted')
        inspect_read(reject)
        assert not log_path.read_text()
        print(json.dumps({'error_is_pure': True}))
    """)
    assert result["error_is_pure"]


def test_snapshot_preserves_missing_named_profile_guard(tmp_path):
    result = run_probe(tmp_path, """
        os.environ['HERMES_HOME'] = str(root / 'profiles' / 'missing')
        def reject():
            try:
                cfg.load_config_snapshot()
            except FileNotFoundError:
                return
            raise AssertionError('missing named profile accepted')
        inspect_read(reject)
        assert not (root / 'profiles').exists()
        print(json.dumps({'guard_preserved': True}))
    """)
    assert result["guard_preserved"]
