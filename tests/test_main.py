"""Tests for the Fire entry point's framework validation boundary."""

import pytest

from rag_against_the_machine import main


def test_invalid_command_uses_framework_diagnostic(monkeypatch, capsys):
    """Typos should be reported by cli_fw before Fire gets control."""
    monkeypatch.setattr(
        main.sys, "argv", ["rag_against_the_machine", "servee"]
    )
    fire_called = False

    def fire_stub():
        nonlocal fire_called
        fire_called = True

    monkeypatch.setattr(main, "run_with_fire", fire_stub)

    with pytest.raises(SystemExit) as exc_info:
        main.main()

    assert exc_info.value.code == 2
    assert fire_called is False
    output = capsys.readouterr().out
    assert "rag_against_the_machine::cli::parser::unknown::argument" in output
    assert "rag_against_the_machine servee" in output
    assert "Unknown command: servee. Did you mean 'serve'?" in output


def test_valid_command_is_still_delegated_to_fire(monkeypatch):
    """Validation must not replace Python Fire as the execution mechanism."""
    monkeypatch.setattr(
        main.sys,
        "argv",
        ["rag_against_the_machine", "serve", "--port", "9000"],
    )
    called = False

    def fire_stub():
        nonlocal called
        called = True

    monkeypatch.setattr(main, "run_with_fire", fire_stub)

    main.main()

    assert called is True


def test_out_of_range_port_is_rejected_before_fire(monkeypatch, capsys):
    """Invalid TCP ports should produce a concise error instead of a bind traceback."""
    monkeypatch.setattr(
        main.sys,
        "argv",
        ["rag_against_the_machine", "serve", "--port", "70000"],
    )
    fire_called = False

    def fire_stub():
        nonlocal fire_called
        fire_called = True

    monkeypatch.setattr(main, "run_with_fire", fire_stub)

    with pytest.raises(SystemExit) as exc_info:
        main.main()

    assert exc_info.value.code == 2
    assert fire_called is False
    assert "--port must be between 0 and 65535" in capsys.readouterr().err


def test_keyboard_interrupt_exits_cleanly(monkeypatch, capsys):
    """Ctrl-C should stop the command without escaping to the entry thread."""
    monkeypatch.setattr(main.sys, "argv", ["rag_against_the_machine", "serve"])

    def interrupted_fire():
        raise KeyboardInterrupt

    monkeypatch.setattr(main, "run_with_fire", interrupted_fire)

    main.main()

    assert "Server stopped. Goodbye!" in capsys.readouterr().out


def test_invalid_subcommand_option_shows_the_complete_command_line(
    monkeypatch, capsys
):
    """Nested parser diagnostics retain the command and all supplied options."""
    monkeypatch.setattr(
        main.sys,
        "argv",
        ["rag_against_the_machine", "serve", "--port", "nope"],
    )

    with pytest.raises(SystemExit):
        main.main()

    output = capsys.readouterr().out
    assert "serve --port nope" in output
