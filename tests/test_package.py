import pytest

import qbq
from qbq.cli import main


def test_version_is_exposed():
    assert qbq.__version__ == "0.1.0"


def test_cli_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "qbq 0.1.0" in capsys.readouterr().out


def test_cli_without_args_prints_help(capsys):
    assert main([]) == 0
    assert "usage: qbq" in capsys.readouterr().out
