# Paper reproductions

This directory ships the run artifacts and orchestration scripts that
produced every numbered table in `paper.tex`. Each per-table README
under `tables/` lists:
- The exact tarball under `archives/` that holds the raw run output
  (HEARTBEAT, training logs, search logs, deployed runtime files).
- The orchestration script under `orchestration_scripts/` that
  reproduces the measurement from scratch on a 7-node trn1.32xlarge
  cluster (or 1-node where relevant).
- A `verify.sh` snippet that extracts the tarball and prints the
  cell values that fed the paper row, so a reviewer can spot-check
  without re-running on hardware.

See `archives_index.md` (one level up) for the full table → archive
mapping with sizes and SHA-256 prefixes.
