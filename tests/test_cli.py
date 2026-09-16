from typer.testing import CliRunner

from amnmt import __version__
from amnmt.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_config_validate() -> None:
    result = runner.invoke(app, ["config", "validate", "configs/tiny.yaml"])
    assert result.exit_code == 0
    assert "tiny" in result.output
