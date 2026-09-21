from unittest.mock import MagicMock, patch

from bias_scope_agent import cli


class TestCliMain:
    def test_exits_cleanly_on_eof(self):
        with (
            patch("bias_scope_agent.cli.AgentLoop"),
            patch("builtins.input", side_effect=EOFError),
        ):
            assert cli.main([]) == 0

    def test_exits_cleanly_on_exit_command(self):
        with (
            patch("bias_scope_agent.cli.AgentLoop"),
            patch("builtins.input", side_effect=["exit"]),
        ):
            assert cli.main([]) == 0

    def test_exits_cleanly_on_quit_command_case_insensitive(self):
        with (
            patch("bias_scope_agent.cli.AgentLoop"),
            patch("builtins.input", side_effect=["QUIT"]),
        ):
            assert cli.main([]) == 0

    def test_runs_a_turn_and_prints_the_reply(self, capsys):
        fake_loop = MagicMock()
        fake_loop.run_turn.return_value = "hi there"
        with (
            patch("bias_scope_agent.cli.AgentLoop", return_value=fake_loop),
            patch("builtins.input", side_effect=["hello", "exit"]),
        ):
            exit_code = cli.main([])
        assert exit_code == 0
        fake_loop.run_turn.assert_called_once_with("hello")
        assert "hi there" in capsys.readouterr().out
