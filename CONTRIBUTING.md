# Contributing

Issues and pull requests are welcome.

## Licensing of contributions

By submitting a contribution you agree that it is licensed under this project's
license (see `LICENSE`), and that **The Simons Foundation, Inc. may relicense it
under any license permitted by the Foundation's intellectual-property policy** —
currently Apache 2.0, GPL, LGPL, MIT, or 3-clause BSD.

That second clause exists so the project can change license later without having to
track down every past contributor for permission. Relicensing is trivial while a
project has one author and becomes progressively harder with each additional one;
this keeps the option open. It does not affect your own rights to your contribution,
and it does not let anyone revoke a license already granted for a published version.

## Practical notes

- Run the test suite before opening a PR (`pytest -q`). CI runs it on every push and
  pull request, and it must be green to merge.
- This package sits between `em-blockrun` and its consumers (`em-seg-morpho`), so a
  change here can affect them; run their suites too if you have them checked out.
