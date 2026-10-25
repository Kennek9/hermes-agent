"""Behavioral regression coverage for the wheel/sdist distribution guard."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _build_artifact(kind: str, tmp_path, *, nix_build: bool, library: bool = False) -> subprocess.CompletedProcess[str]:
    """Invoke the real PEP 517 hook (build_sdist / build_wheel) as a subprocess.

    The wheel and sdist guards live in SEPARATE cmdclass entries in setup.py
    (the bdist_wheel one behind a try/except ImportError), so each hook needs
    its own regression coverage — a passing sdist test proves nothing about
    the wheel path.
    """
    env = os.environ.copy()
    # nix develop exports this too, so it must not grant permission to build
    # a distributable artifact.
    env["NIX_BUILD_TOP"] = "/build/devshell"
    if nix_build:
        env["HERMES_NIX_BUILD"] = "1"
    else:
        env.pop("HERMES_NIX_BUILD", None)
    # Redirect setuptools' scratch dirs (build/, *.egg-info) into tmp_path so
    # the allowed-marker build doesn't litter the real worktree.
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    extra_cfg = tmp_path / "dist-extra.cfg"
    extra_cfg.write_text(
        f"[build]\nbuild_base = {scratch / 'build'}\n\n[egg_info]\negg_base = {scratch}\n",
        encoding="utf-8",
    )
    env["DIST_EXTRA_CONFIG"] = str(extra_cfg)
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "from setuptools.build_meta import build_{kind}; build_{kind}(r'{out}', config_settings={settings!r})".format(
                kind=kind, out=tmp_path, settings={"--build-option": ["--library"]} if library else None,
            ),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize("kind", ["sdist", "wheel"])
def test_artifact_build_rejects_nix_development_shell_environment(kind, tmp_path):
    result = _build_artifact(kind, tmp_path, nix_build=False)

    assert result.returncode != 0
    assert "Building wheels or sdists for hermes-agent is not supported" in result.stderr


def test_explicit_library_wheel_contains_execution_resources(tmp_path):
    import zipfile

    result = _build_artifact("wheel", tmp_path, nix_build=False, library=True)
    assert result.returncode == 0, result.stderr[-3000:]
    wheel = next(tmp_path.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    assert {"run_agent.py", "hermes_runtime.py", "hermes_platform/resources.py"} <= names
    for directory in ("skills", "optional-skills", "optional-mcps", "locales", "assets"):
        assert any(name.startswith(f"hermes_resources/{directory}/") for name in names), directory


