"""Tests for enerdatasynth.cli -- specifically the `--resource {pjm_load,wind}`
dispatch added to `generate`/`evaluate`/`report`. The pjm_load-only commands
(`preprocess`/`train`/`sweep`/`calibrate`/`pipeline`) and pjm_load's own
generate/evaluate/report bodies are pre-existing and covered by their
underlying modules' own tests; this file only exercises the new branching
logic and the parser shape, so a wind resource stays reachable from the CLI
without regressing the pjm_load default path."""

from __future__ import annotations

import pytest

from enerdatasynth import cli


def test_build_parser_has_all_eight_subcommands():
    parser = cli.build_parser()
    subparsers_action = next(
        a for a in parser._actions if isinstance(a, __import__("argparse")._SubParsersAction)
    )
    assert set(subparsers_action.choices) == {
        "preprocess", "train", "sweep", "generate",
        "calibrate", "evaluate", "report", "pipeline",
    }


def test_generate_defaults_to_pjm_load_resource():
    parser = cli.build_parser()
    args = parser.parse_args(["generate", "--diffusion-steps", "100", "--hidden-dim", "128", "--n-layers", "4"])
    assert args.resource == "pjm_load"


def test_evaluate_run_id_defaults_to_none_not_required():
    # pjm_load's requiredness is enforced manually in cmd_evaluate, not by
    # argparse, so --resource wind (which needs --preset instead) can share
    # the subparser without --run-id being mandatory at parse time.
    parser = cli.build_parser()
    args = parser.parse_args(["evaluate", "--resource", "wind", "--preset", "1ch"])
    assert args.run_id is None
    assert args.preset == "1ch"


def test_cmd_evaluate_pjm_load_without_run_id_raises_clean_system_exit():
    parser = cli.build_parser()
    args = parser.parse_args(["evaluate"])
    with pytest.raises(SystemExit, match="--run-id is required"):
        cli.cmd_evaluate(args)


def test_cmd_generate_wind_dispatches_and_produces_real_profile(capsys):
    parser = cli.build_parser()
    args = parser.parse_args([
        "generate", "--resource", "wind", "--preset", "1ch",
        "--wind-regime", "mid", "--n-realizations", "1", "--seed", "0",
    ])
    cli.cmd_generate(args)
    out = capsys.readouterr().out
    assert "Generated 1 realizations" in out
    assert "preset=1ch" in out
    assert "regime=mid" in out


def test_cmd_evaluate_wind_dispatches_and_writes_real_report(tmp_path, monkeypatch, capsys):
    import enerdatasynth.resources.wind.evaluation.failure_analysis as fa_module

    monkeypatch.setattr(fa_module, "DATA_ROOT", tmp_path)
    parser = cli.build_parser()
    args = parser.parse_args(["evaluate", "--resource", "wind", "--preset", "1ch"])
    cli.cmd_evaluate(args)
    out = capsys.readouterr().out
    assert "Issues detected" in out
    assert (tmp_path / "reports" / "wind" / "failure_analysis_1ch.md").exists()


def test_cmd_report_wind_dispatches_and_writes_real_summary(tmp_path, monkeypatch, capsys):
    import enerdatasynth.resources.wind.evaluation.results_report as rr_module

    monkeypatch.setattr(rr_module, "DATA_ROOT", tmp_path)
    parser = cli.build_parser()
    args = parser.parse_args(["report", "--resource", "wind"])
    cli.cmd_report(args)
    out = capsys.readouterr().out
    saved_path = tmp_path / "reports" / "wind" / "results_summary.md"
    assert str(saved_path) in out
    assert saved_path.exists()


def test_report_defaults_to_pjm_load_resource():
    parser = cli.build_parser()
    args = parser.parse_args(["report"])
    assert args.resource == "pjm_load"


def test_cmd_calibrate_passes_seed_through(monkeypatch):
    # `calibrate` used to accept --seed and silently drop it, making the
    # fitted offset irreproducible from the command line
    import enerdatasynth.resources.pjm_load.evaluation.calibration as cal

    seen = {}

    def fake_calibrate_checkpoint(**kwargs):
        seen.update(kwargs)
        return {"scalar": 0.0, "path": "x"}

    monkeypatch.setattr(cal, "calibrate_checkpoint", fake_calibrate_checkpoint)
    args = cli.build_parser().parse_args(["calibrate", "--run-id", "r", "--seed", "42"])
    cli.cmd_calibrate(args)
    assert seen["seed"] == 42
    assert seen["run_id"] == "r"
