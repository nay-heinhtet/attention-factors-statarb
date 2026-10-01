"""Command-line entry point: ``python -m statarb <command> [options]``.

Each command maps to a ``module:function`` target that is called as ``function(cfg, args)``.
A command whose module doesn't exist yet reports the PLAN.md phase that builds it.
"""

from __future__ import annotations

import argparse
import importlib
import sys

from statarb.config import load_config
from statarb.utils import setup_logging

STRATEGIES = ["attention", "pca", "pairs", "reversal"]

# command -> (target "module:function", PLAN.md phase, help text)
COMMANDS: dict[str, tuple[str, int, str]] = {
    "membership": ("statarb.data.membership:run", 1, "download point-in-time S&P 500 membership, build spells"),
    "yahoo": ("statarb.data.yahoo:run", 1, "download daily prices from Yahoo (cached, resumable)"),
    "resolve": ("statarb.data.resolve:run", 1, "map membership spells to price series"),
    "tiingo": ("statarb.data.tiingo:run", 1, "download delisted names from Tiingo (slow, resumable)"),
    "sectors": ("statarb.data.sectors:run", 1, "build the sector map"),
    "panel": ("statarb.data.panel:run", 2, "build the aligned price/return panel"),
    "universe": ("statarb.data.universe:run", 2, "build the monthly top-200 universe"),
    "coverage": ("statarb.data.coverage:run", 2, "write the survivorship coverage report"),
    "features": ("statarb.features:run", 3, "compute the 15 characteristics"),
    "tensors": ("statarb.tensors:run", 5, "build model-ready tensors"),
    "tune": ("statarb.walkforward:tune", 6, "tune on 2011-2016, validate on 2017-2018"),
    "freeze": ("statarb.walkforward:freeze", 6, "write configs/frozen.yaml from the tuning results"),
    "oos": ("statarb.walkforward:oos", 7, "run the out-of-sample walk-forward"),
    "checks": ("statarb.bias_checks:run", 8, "run the look-ahead and survivorship checks"),
    "report": ("statarb.report:run", 10, "build the report tables and figures"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m statarb", description=__doc__.splitlines()[0])
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")
    cmd = {name: sub.add_parser(name, help=help_text) for name, (_, _, help_text) in COMMANDS.items()}

    cmd["universe"].add_argument("--kind", choices=["pit", "survivor"], default="pit")
    cmd["tensors"].add_argument("--universe", choices=["pit", "survivor"], default="pit")
    cmd["tensors"].add_argument("--variant", choices=["real", "noise"], default="real")
    cmd["tune"].add_argument("--strategy", choices=STRATEGIES, required=True)

    oos = cmd["oos"]
    oos.add_argument("--strategy", choices=STRATEGIES, required=True)
    oos.add_argument("--universe", choices=["pit", "survivor"], default="pit")
    oos.add_argument("--sequence", choices=["lstm", "longconv"], help="ablation: override model.sequence")
    oos.add_argument("--drop-group", help="ablation: drop one feature group")
    oos.add_argument("--K", type=int, help="ablation: override the number of factors")
    oos.add_argument("--exec-lag", type=int, choices=[0, 1], help="override backtest.exec_lag")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    target, phase, _ = COMMANDS[args.command]
    module_name, func_name = target.split(":")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        print(f"`statarb {args.command}` is built in Phase {phase} and isn't implemented yet.", file=sys.stderr)
        return 2
    getattr(module, func_name)(load_config(), args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
