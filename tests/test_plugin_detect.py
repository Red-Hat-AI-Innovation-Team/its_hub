"""Tests for the plugin detection script (scripts/its_detect.sh)."""

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "its_detect.sh"


def run_detect(env_override=None, cwd=None):
    """Run its_detect.sh and return parsed key-value output."""
    env = os.environ.copy()
    if env_override:
        env.update(env_override)
    result = subprocess.run(
        [str(SCRIPT_PATH)],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd or Path(__file__).parent.parent,
    )
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    lines = [line for line in result.stdout.strip().split("\n") if "=" in line]
    return dict(line.split("=", 1) for line in lines)


class TestDetectScript:
    """Test the environment detection script output format and behavior."""

    def test_output_has_all_required_keys(self):
        """Detection script must output exactly four keys."""
        output = run_detect()
        assert "server" in output
        assert "library" in output
        assert "installer" in output
        assert "config" in output

    def test_server_values_are_valid(self):
        """Server status must be 'running' or 'stopped'."""
        output = run_detect()
        assert output["server"] in ("running", "stopped")

    def test_library_values_are_valid(self):
        """Library status must be 'installed' or 'missing'."""
        output = run_detect()
        assert output["library"] in ("installed", "missing")

    def test_installer_values_are_valid(self):
        """Installer must be 'uv', 'pip', or 'none'."""
        output = run_detect()
        assert output["installer"] in ("uv", "pip", "none")

    def test_config_values_are_valid(self):
        """Config status must be 'found' or 'missing'."""
        output = run_detect()
        assert output["config"] in ("found", "missing")

    def test_config_missing_when_no_config_file(self, tmp_path):
        """Config should be 'missing' when pointing to nonexistent path."""
        output = run_detect(
            env_override={"ITS_HUB_CONFIG": str(tmp_path / "nonexistent.json")}
        )
        assert output["config"] == "missing"

    def test_config_found_when_config_exists(self, tmp_path):
        """Config should be 'found' when the config file exists."""
        config_file = tmp_path / "config.json"
        config_file.write_text('{"iaas_port": 8108}')
        output = run_detect(env_override={"ITS_HUB_CONFIG": str(config_file)})
        assert output["config"] == "found"

    def test_server_stopped_when_no_server(self):
        """Server should be 'stopped' when nothing is listening."""
        output = run_detect(
            env_override={"ITS_HUB_CONFIG": "/dev/null/nonexistent"}
        )
        assert output["server"] == "stopped"

    def test_library_installed_in_dev_environment(self):
        """In the dev environment, its_hub should be installed."""
        output = run_detect()
        assert output["library"] == "installed"

    def test_output_is_exactly_four_lines(self):
        """Script should output exactly four key=value lines."""
        result = subprocess.run(
            [str(SCRIPT_PATH)],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        lines = [
            line for line in result.stdout.strip().split("\n") if line.strip()
        ]
        assert len(lines) == 4
