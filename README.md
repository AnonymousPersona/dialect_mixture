# Replication code

Requirements

- R >= 4.4
- CmdStan
- cmdstanr

Installation

install.packages(c(
  "cmdstanr",
  "posterior",
  "loo",
  ...
))

Run

Rscript r/repo_models/01_run_bb_tg_ctx_stgp_ctx_stz.R

Outputs are written to

results/