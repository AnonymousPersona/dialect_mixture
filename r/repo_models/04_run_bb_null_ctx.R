# ==========================================================
#  04_run_bb_null_ctx.R
#
#  Null model: intercept + EIR shift only (alpha, gamma_ctx,
#  phi).  No date trend, no GP, no type effects.  Provides
#  a baseline LOO reference for all other models.
#
#  Requires: 00_setup.R (provides dat_stgp_ctx, has_var)
# ==========================================================

source(here::here("r", "repo_models", "00_setup.R"))

# ----------------------------------------------------------
#  PATHS
# ----------------------------------------------------------

fit_name_null <- "04_bb_null_ctx"
out_dir_null  <- file.path(results_root, fit_name_null)
dir.create(out_dir_null, recursive = TRUE, showWarnings = FALSE)

csv_dir_null <- file.path(out_dir_null, "csv")
dir.create(csv_dir_null, recursive = TRUE, showWarnings = FALSE)

# ----------------------------------------------------------
#  DATA — null model needs only N, y1, n1, y2, n2
# ----------------------------------------------------------

dat_null <- list(
  N  = dat_stgp_ctx$N,
  y1 = dat_stgp_ctx$y1,
  n1 = dat_stgp_ctx$n1,
  y2 = dat_stgp_ctx$y2,
  n2 = dat_stgp_ctx$n2
)

stopifnot(
  lengths(dat_null[c("y1", "n1", "y2", "n2")]) == dat_null$N,
  all(dat_null$y1 <= dat_null$n1),
  all(dat_null$y2 <= dat_null$n2)
)

message(sprintf("Null model data: N = %d inscriptions.", dat_null$N))

# ----------------------------------------------------------
#  COMPILE & SAMPLE
# ----------------------------------------------------------

stan_file_null <- file.path(
  here::here("stan"),
  paste0(fit_name_null, ".stan")
)
if (!file.exists(path.expand(stan_file_null)))
  stop("Stan file not found: ", stan_file_null)
mod_null <- cmdstanr::cmdstan_model(stan_file_null)

fit_bb_null_ctx <- mod_null$sample(
  data             = dat_null,
  output_dir       = csv_dir_null,
  seed             = 2025,
  chains           = 4, parallel_chains = 4,
  iter_warmup      = 2000, iter_sampling  = 2000,
  adapt_delta      = 0.90, max_treedepth  = 10,
  refresh          = 50
)

# ----------------------------------------------------------
#  LOO
# ----------------------------------------------------------

if (has_var(fit_bb_null_ctx, "log_lik")) {
  ll_mat_null <- fit_bb_null_ctx$draws("log_lik", format = "matrix")

  if (is.matrix(ll_mat_null) && ncol(ll_mat_null) > 20000) {
    set.seed(1); keep_null <- sort(sample.int(ncol(ll_mat_null), 20000))
    ll_mat_null <- ll_mat_null[, keep_null, drop = FALSE]
  }

  if (is.matrix(ll_mat_null) && ncol(ll_mat_null) > 0) {
    loo_obj_null <- loo::loo(ll_mat_null, save_psis = TRUE, moment_match = FALSE)

    est_null <- loo_obj_null$estimates
    writeLines(paste0(
      "elpd_loo = ", sprintf("%.3f", est_null["elpd_loo", "Estimate"]),
      " (SE = ", sprintf("%.3f", est_null["elpd_loo", "SE"]), ")\n",
      "p_loo    = ", sprintf("%.3f", est_null["p_loo",    "Estimate"]), "\n",
      "looic    = ", sprintf("%.3f", est_null["looic",     "Estimate"]), "\n",
      "---------\n",
      "Pareto k diagnostics:\n",
      paste(capture.output(print(loo::pareto_k_table(loo_obj_null))),
            collapse = "\n"), "\n"
    ), file.path(out_dir_null, "loo.txt"))

    readr::write_csv(
      tibble::tibble(
        metric   = rownames(est_null),
        estimate = as.numeric(est_null[, "Estimate"]),
        se       = as.numeric(est_null[, "SE"])
      ),
      file.path(out_dir_null, "loo_estimates.csv")
    )
    pk_null <- loo::pareto_k_table(loo_obj_null)
    suppressWarnings(
      readr::write_csv(as.data.frame(pk_null),
                       file.path(out_dir_null, "pareto_k.csv"))
    )
    saveRDS(loo_obj_null, file.path(out_dir_null, "loo.rds"))
  } else {
    readr::write_lines("Found 'log_lik' but it contained 0 columns.",
                       file.path(out_dir_null, "loo.txt"))
  }
} else {
  readr::write_lines("log_lik not found in fitted draws; LOO not computed.",
                     file.path(out_dir_null, "loo.txt"))
}

# ----------------------------------------------------------
#  SAVE FIT OBJECT
# ----------------------------------------------------------

saveRDS(fit_bb_null_ctx,
        file.path(out_dir_null, paste0(fit_name_null, ".rds")))

# ----------------------------------------------------------
#  PARAMETER SUMMARIES
# ----------------------------------------------------------

summ_all_null <- fit_bb_null_ctx$summary()
readr::write_csv(as.data.frame(summ_all_null),
                 file.path(out_dir_null, "summary_all.csv"))

core_vars_null <- c("alpha", "gamma_ctx", "phi")
summ_core_null <- tryCatch(
  fit_bb_null_ctx$summary(variables = core_vars_null),
  error = function(e) tibble::tibble(note = conditionMessage(e))
)
readr::write_csv(as.data.frame(summ_core_null),
                 file.path(out_dir_null, "summary_core.csv"))

# ---- theta_bar quantities ----
for (qty in c("theta_bar", "theta_bar_out", "theta_bar_in")) {
  if (has_var(fit_bb_null_ctx, qty)) {
    fit_bb_null_ctx$summary(variables = qty) |>
      dplyr::mutate(
        row = as.integer(gsub(paste0("^", qty, "\\[(\\d+)\\]$"), "\\1", variable))
      ) |>
      dplyr::relocate(row, .before = 1) |>
      (\(x) readr::write_csv(as.data.frame(x),
                              file.path(out_dir_null,
                                        paste0("summary_", qty, ".csv"))))()
  }
}

# ----------------------------------------------------------
#  DIAGNOSTICS
# ----------------------------------------------------------

diag_df_null <- tryCatch(
  as.data.frame(fit_bb_null_ctx$diagnostic_summary()),
  error = function(e) NULL
)
if (!is.null(diag_df_null))
  readr::write_csv(diag_df_null,
                   file.path(out_dir_null, "diagnostic_summary.csv"))

# ----------------------------------------------------------
#  PROVENANCE
# ----------------------------------------------------------

readr::write_file(
  jsonlite::toJSON(list(
    model_file    = stan_file_null,
    fit_name      = fit_name_null,
    results_dir   = normalizePath(out_dir_null),
    N             = dat_null$N,
    chains        = 4L,
    iter_warmup   = 2000L,
    iter_sampling = 2000L,
    adapt_delta   = 0.90,
    seed          = 2025L
  ), auto_unbox = TRUE, pretty = TRUE),
  file.path(out_dir_null, "run_metadata.json")
)

# ----------------------------------------------------------
#  CONSOLE SUMMARY
# ----------------------------------------------------------

message("\nNull model — core parameters:")
print(tibble::as_tibble(summ_core_null))
