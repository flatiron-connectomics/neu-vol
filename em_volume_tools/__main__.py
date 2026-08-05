"""``python -m em_volume_tools`` — the same entry point as the ``em-vol`` command.

Worth having as well as the console script: ``python -m`` works from a source checkout
without the package being installed, and makes it unambiguous which interpreter is
running when several environments are on PATH.
"""

from em_volume_tools.cli import main

raise SystemExit(main())
