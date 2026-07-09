# ==========================================================
#  01_run_bb_tg_ctx_stgp_ctx_stz.R
#
#  Base STGP model with global EIR shift (gamma_ctx) and
#  spatiotemporal GP intercept surface (f_site).
#  This is the primary reference model for all comparisons.
# ==========================================================

source(here::here("r", "repo_models", "00_setup.R"))

# ----------------------------------------------------------
#  PATHS
# ----------------------------------------------------------

fit_name <- "01_bb_tg_ctx_stgp_ctx_stz"
out_dir  <- file.path(results_root, fit_name)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

csv_dir <- file.path(out_dir, "csv")
dir.create(csv_dir, recursive = TRUE, showWarnings = FALSE)

# ----------------------------------------------------------
#  COMPILE & SAMPLE
# ----------------------------------------------------------

stan_file <- file.path(
  here::here("stan"),
  paste0(fit_name, ".stan")
)
if (!file.exists(path.expand(stan_file))) stop("Stan file not found: ", stan_file)
mod <- cmdstanr::cmdstan_model(stan_file)

fit_bb_tg_ctx_stgp_ctx_stz <- mod$sample(
  data             = dat_stgp_ctx,
  output_dir       = csv_dir,
  seed             = 2025,
  chains           = 4, parallel_chains = 4,
  iter_warmup      = 2000, iter_sampling  = 2000,
  adapt_delta      = 0.95, max_treedepth  = 12,
  refresh          = 50
)

# ----------------------------------------------------------
#  LOO
# ----------------------------------------------------------

if (has_var(fit_bb_tg_ctx_stgp_ctx_stz, "log_lik")) {
  ll_mat <- fit_bb_tg_ctx_stgp_ctx_stz$draws("log_lik", format = "matrix")

  if (is.matrix(ll_mat) && ncol(ll_mat) > 20000) {
    set.seed(1); keep <- sort(sample.int(ncol(ll_mat), 20000))
    ll_mat <- ll_mat[, keep, drop = FALSE]
  }

  if (is.matrix(ll_mat) && ncol(ll_mat) > 0) {
    loo_obj <- loo::loo(ll_mat, save_psis = TRUE, moment_match = FALSE)

    est <- loo_obj$estimates
    writeLines(paste0(
      "elpd_loo = ", sprintf("%.3f", est["elpd_loo", "Estimate"]),
      " (SE = ", sprintf("%.3f", est["elpd_loo", "SE"]), ")\n",
      "p_loo    = ", sprintf("%.3f", est["p_loo",    "Estimate"]), "\n",
      "looic    = ", sprintf("%.3f", est["looic",     "Estimate"]), "\n",
      "---------\n",
      "Pareto k diagnostics:\n",
      paste(capture.output(print(loo::pareto_k_table(loo_obj))), collapse = "\n"), "\n"
    ), file.path(out_dir, "loo.txt"))

    readr::write_csv(
      tibble::tibble(
        metric   = rownames(est),
        estimate = as.numeric(est[, "Estimate"]),
        se       = as.numeric(est[, "SE"])
      ),
      file.path(out_dir, "loo_estimates.csv")
    )
    pk <- loo::pareto_k_table(loo_obj)
    suppressWarnings(
      readr::write_csv(as.data.frame(pk), file.path(out_dir, "pareto_k.csv"))
    )
    saveRDS(loo_obj, file.path(out_dir, "loo.rds"))
  } else {
    readr::write_lines("Found 'log_lik' but it contained 0 columns.",
                       file.path(out_dir, "loo.txt"))
  }
} else {
  readr::write_lines("log_lik not found in fitted draws; LOO not computed.",
                     file.path(out_dir, "loo.txt"))
}

# ----------------------------------------------------------
#  SAVE FIT OBJECT
# ----------------------------------------------------------

saveRDS(fit_bb_tg_ctx_stgp_ctx_stz,
        file.path(out_dir, paste0(fit_name, ".rds")))

# ----------------------------------------------------------
#  INDEX TABLES
# ----------------------------------------------------------

tibble::tibble(type_id = seq_along(type_levels), type = type_levels) |>
  readr::write_csv(file.path(out_dir, "type_index.csv"))

sites_tbl |>
  dplyr::mutate(site_id = dplyr::row_number()) |>
  readr::write_csv(file.path(out_dir, "site_index.csv"))

tibble::tibble(
  t0_year       = time_info$t0,
  t_scale_years = time_info$scale_years,
  K_u           = dat_stgp_ctx$K_u,
  coord_tol     = time_info$tol
) |>
  readr::write_csv(file.path(out_dir, "time_config.csv"))

# ----------------------------------------------------------
#  PARAMETER SUMMARIES
# ----------------------------------------------------------

summ_all <- fit_bb_tg_ctx_stgp_ctx_stz$summary()
readr::write_csv(summ_all, file.path(out_dir, "summary_all.csv"))

core_vars <- c("alpha", "beta_date", "gamma_ctx", "phi",
               "sigma_gp", "rho_s", "rho_t")
summ_core <- tryCatch(
  fit_bb_tg_ctx_stgp_ctx_stz$summary(variables = core_vars),
  error = function(e) tibble::tibble(note = conditionMessage(e))
)
readr::write_csv(summ_core, file.path(out_dir, "summary_core.csv"))

# ---- Type fixed effects (sum-to-zero; beta_type[1..T]) ----
type_draws <- tryCatch(
  posterior::as_draws_df(
    fit_bb_tg_ctx_stgp_ctx_stz$draws(variables = "beta_type")
  ),
  error = function(e) NULL
)
if (!is.null(type_draws)) {
  bt_cols <- grep("^beta_type\\[\\d+\\]$", names(type_draws), value = TRUE)
  bt_cols <- bt_cols[order(as.integer(gsub(".*\\[(\\d+)\\]$", "\\1", bt_cols)))]
  summ_type <- tibble::tibble(
    variable = bt_cols,
    type_id  = seq_along(bt_cols),
    type     = type_levels[seq_along(bt_cols)],
    median   = sapply(bt_cols, \(nm) stats::median(type_draws[[nm]])),
    lo95     = sapply(bt_cols, \(nm) stats::quantile(type_draws[[nm]], 0.025)),
    hi95     = sapply(bt_cols, \(nm) stats::quantile(type_draws[[nm]], 0.975))
  )
  readr::write_csv(summ_type, file.path(out_dir, "summary_beta_type_fixed.csv"))
} else {
  message("beta_type not found in draws; summary_beta_type_fixed.csv not written.")
}

# ---- f_site: baseline dialect GP surface ----
if (has_var(fit_bb_tg_ctx_stgp_ctx_stz, "f_site")) {
  fit_bb_tg_ctx_stgp_ctx_stz$summary(variables = "f_site") |>
    dplyr::mutate(
      site_id = as.integer(gsub("^f_site\\[(\\d+)\\]$", "\\1", variable))
    ) |>
    dplyr::left_join(
      sites_tbl |> dplyr::mutate(site_id = dplyr::row_number()),
      by = "site_id"
    ) |>
    dplyr::relocate(site_id, latitude_site, longitude_site, .after = variable) |>
    readr::write_csv(file.path(out_dir, "summary_f_site.csv"))
}

# ---- theta_bar quantities ----
for (qty in c("theta_bar", "theta_bar_out", "theta_bar_in")) {
  if (has_var(fit_bb_tg_ctx_stgp_ctx_stz, qty)) {
    fit_bb_tg_ctx_stgp_ctx_stz$summary(variables = qty) |>
      dplyr::mutate(
        row = as.integer(gsub(paste0("^", qty, "\\[(\\d+)\\]$"), "\\1", variable))
      ) |>
      dplyr::relocate(row, .before = 1) |>
      readr::write_csv(file.path(out_dir, paste0("summary_", qty, ".csv")))
  }
}

# ----------------------------------------------------------
#  DIAGNOSTICS
# ----------------------------------------------------------

diag_df <- tryCatch(
  as.data.frame(fit_bb_tg_ctx_stgp_ctx_stz$diagnostic_summary()),
  error = function(e) NULL
)
if (!is.null(diag_df))
  readr::write_csv(diag_df, file.path(out_dir, "diagnostic_summary.csv"))

# ----------------------------------------------------------
#  PROVENANCE
# ----------------------------------------------------------

readr::write_file(
  jsonlite::toJSON(list(
    model_file    = stan_file,
    fit_name      = fit_name,
    results_dir   = normalizePath(out_dir),
    N             = dat_stgp_ctx$N,
    S             = dat_stgp_ctx$S,
    T             = dat_stgp_ctx$T,
    K_u           = dat_stgp_ctx$K_u,
    chains        = 4L,
    iter_warmup   = 2000L,
    iter_sampling = 2000L,
    adapt_delta   = 0.95,
    seed          = 2025L
  ), auto_unbox = TRUE, pretty = TRUE),
  file.path(out_dir, "run_metadata.json")
)

# ----------------------------------------------------------
#  CONSOLE SUMMARY
# ----------------------------------------------------------

message("\nKey scalar parameters:")
print(tibble::as_tibble(summ_core))

message("\nType fixed effects (sum-to-zero beta_type):")
if (exists("summ_type")) print(summ_type) else message("Not available.")

message("\nGP hyperparameters:")
print(tibble::as_tibble(
  fit_bb_tg_ctx_stgp_ctx_stz$summary(
    variables = c("sigma_gp", "rho_s", "rho_t")
  )
))
