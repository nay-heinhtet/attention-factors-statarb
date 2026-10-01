import pytest

from statarb.__main__ import COMMANDS, main


def test_help_lists_commands(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "membership" in capsys.readouterr().out


def test_unbuilt_command_reports_its_phase(monkeypatch, capsys):
    monkeypatch.setitem(COMMANDS, "report", ("statarb.not_built_yet:run", 10, "test"))
    assert main(["report"]) == 2
    assert "Phase 10" in capsys.readouterr().err
