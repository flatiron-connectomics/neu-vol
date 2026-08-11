"""What the documentation build depends on, pinned here so it cannot quietly break.

The docs site renders the CLI reference from the real ``ArgumentParser``, which is what
stops published usage from drifting away from ``--help``. Two things have to hold for
that: the parser must be reachable without running it, and importing the module must not
require the conda-only half of the environment — otherwise the GitHub Actions job needs
micromamba and flyem-forge instead of a plain pip install.
"""

import argparse
import subprocess
import sys

# conda-only on flyem-forge, or heavy enough that pulling them in would change the docs
# job from "pip install" to "build a conda environment"
CONDA_ONLY = ["vol2mesh", "dvidutils", "kimimaro", "DracoPy", "osteoid", "tensorstore",
              "h5py"]


def test_build_parser_returns_the_parser_without_running_it():
    from em_volume_tools.cli import build_parser

    parser = build_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    assert parser.prog == "em-vol"


def test_every_subcommand_is_reachable_from_the_parser():
    """The cheat sheet and reference are built by walking this; a subcommand that is not
    here is a subcommand that silently never gets documented."""
    from em_volume_tools import cli

    parser = cli.build_parser()
    subs = next(a for a in parser._actions
                if isinstance(a, argparse._SubParsersAction)).choices
    assert set(subs) == {"info", "convert", "downsample", "create", "write",
                         "progress", "bboxes-json", "relabel", "ng-url-gen"}
    for name, sub in subs.items():
        assert sub.format_usage().strip(), f"{name} renders no usage line"


def test_parse_args_still_goes_through_build_parser():
    """Splitting the two must not let the built parser and the used one diverge."""
    from em_volume_tools import cli

    args = cli._parse_args(["info", "somewhere"])
    assert args.func is cli.cmd_info and args.volume == "somewhere"


def test_importing_the_cli_needs_no_conda_only_package():
    """Run in a subprocess: this test session has already imported half of them."""
    code = (
        "import sys; import em_volume_tools.cli; "
        f"print(','.join(m for m in {CONDA_ONLY!r} "
        "if any(k == m or k.startswith(m + '.') for k in sys.modules)))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True).stdout.strip()
    assert out == "", (
        f"importing em_volume_tools.cli now pulls in {out}. The docs build installs "
        f"from PyPI only; these are conda-only or heavy, so this would break it. Defer "
        f"the import into the function that needs it.")
