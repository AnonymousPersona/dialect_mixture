# ==========================================================
#  07_run_bb_tg_ctx_stz.R
#
#  Date-only model: alpha, beta_date, beta_type, gamma_ctx,
#  phi.  No site structure or GP; quantifies the contribution
#  of geography over and above the temporal koine trend.
#
#  Requires: 00_setup.R (provides dat_stgp_ctx, type_levels,
#            time_info, has_var)
# ==========================================================

source(here::here("r", "repo_models", "00_setup.R"))

# ==========================================================
#       bb_tg_ctx_stz  —  run + save
#
#  Date-only / no-GP model: alpha, beta_date, beta_type,
#  gamma_ctx, phi.  No site structure; dat_stgp_ctx is
#  subsetted to the fields the Stan data block accepts.
# ==========================================================

stopifnot(
  exists("dat_stgp_ctx"),
  exists("has_var"),
  exists("type_levels"),
  exists("time_info")
)

# ---------- paths ----------
fit_name_do     <- "07_bb_tg_ctx_stz"
out_dir_do      <- file.path(results_root, fit_name_do)
dir.create(out_dir_do, recursive = TRUE, showWarnings = FALSE)

csv_dir_do <- file.path(out_dir_do, "csv")
dir.create(csv_dir_do, recursive = TRUE, showWarnings = FALSE)

# ==========================================================
#   DATA — subset dat_stgp_ctx to the fields accepted by
#   the date-only Stan data block.  Site structure
#   (S, site_id, latitude_site, longitude_site) is absent
#   from bb_tg_ctx_stz.stan and must not be passed.
# ==========================================================

dat_do <- list(
  N         = dat_stgp_ctx$N,
  y1        = dat_stgp_ctx$y1,
  n1        = dat_stgp_ctx$n1,
  y2        = dat_stgp_ctx$y2,
  n2        = dat_stgp_ctx$n2,
  K_u       = dat_stgp_ctx$K_u,
  u_grid    = dat_stgp_ctx$u_grid,
  t_min_std = dat_stgp_ctx$t_min_std,
  t_max_std = dat_stgp_ctx$t_max_std,
  T         = dat_stgp_ctx$T,
  type      = dat_stgp_ctx$type
)

# Integrity checks
stopifnot(
  lengths(dat_do[c("y1","n1","y2","n2",
                    "t_min_std","t_max_std","type")]) == dat_do$N,
  all(dat_do$type >= 1 & dat_do$type <= dat_do$T),
  length(dat_do$u_grid) == dat_do$K_u
)

message(sprintf(
  "Data assembled: N=%d inscriptions, T=%d types, K_u=%d quadrature nodes.",
  dat_do$N, dat_do$T, dat_do$K_u
))

# ==========================================================
#   COMPILE & SAMPLE
# ==========================================================

stan_file_do <- file.path(
  here::here("stan"),
  paste0(fit_name_do, ".stan")
)
if (!file.exists(path.expand(stan_file_do))) stop("Stan file not found: ", stan_file_do)
mod_do <- cmdstanr::cmdstan_model(stan_file_do)

fit_bb_tg_ctx_stz <- mod_do$sample(
  data             = dat_do,
  output_dir       = csv_dir_do,
  seed             = 2025,
  chains           = 4, parallel_chains = 4,
  iter_warmup      = 2000, iter_sampling  = 2000,
  adapt_delta      = 0.90, max_treedepth  = 10,
  refresh          = 50
)

# ==========================================================
#   SAVE: summaries, LOO, indices, metadata
# ==========================================================

# ---- LOO ----
if (has_var(fit_bb_tg_ctx_stz, "log_lik")) {
  ll_mat_do <- fit_bb_tg_ctx_stz$draws("log_lik", format = "matrix")

  if (is.matrix(ll_mat_do) && ncol(ll_mat_do) > 20000) {
    set.seed(1); keep_do <- sort(sample.int(ncol(ll_mat_do), 20000))
    ll_mat_do <- ll_mat_do[, keep_do, drop = FALSE]
  }

  if (is.matrix(ll_mat_do) && ncol(ll_mat_do) > 0) {
    loo_obj_do <- loo::loo(ll_mat_do, save_psis = TRUE, moment_match = FALSE)

    est_do <- loo_obj_do$estimates
    txt_do <- paste0(
      "elpd_loo = ", sprintf("%.3f", est_do["elpd_loo", "Estimate"]),
      " (SE = ", sprintf("%.3f", est_do["elpd_loo", "SE"]), ")\n",
      "p_loo    = ", sprintf("%.3f", est_do["p_loo",    "Estimate"]), "\n",
      "looic    = ", sprintf("%.3f", est_do["looic",     "Estimate"]), "\n",
      "---------\n",
      "Pareto k diagnostics:\n",
      paste(capture.output(print(loo::pareto_k_table(loo_obj_do))), collapse = "\n"), "\n"
    )
    writeLines(txt_do, file.path(out_dir_do, "loo.txt"))

    readr::write_csv(
      tibble::tibble(
        metric   = rownames(est_do),
        estimate = as.numeric(est_do[, "Estimate"]),
        se       = as.numeric(est_do[, "SE"])
      ),
      file.path(out_dir_do, "loo_estimates.csv")
    )

    pk_do <- loo::pareto_k_table(loo_obj_do)
    suppressWarnings(
      readr::write_csv(as.data.frame(pk_do), file.path(out_dir_do, "pareto_k.csv"))
    )

    saveRDS(loo_obj_do, file.path(out_dir_do, "loo.rds"))
  } else {
    readr::write_lines(
      "Found 'log_lik' but it contained 0 columns.",
      file.path(out_dir_do, "loo.txt")
    )
  }
} else {
  readr::write_lines(
    "log_lik not found in fitted draws; LOO not computed.",
    file.path(out_dir_do, "loo.txt")
  )
}

# ---- Save fit object ----
saveRDS(fit_bb_tg_ctx_stz, file.path(out_dir_do, paste0(fit_name_do, ".rds")))

# ---- Index tables (type only; no site structure in this model) ----
tibble::tibble(type_id = seq_along(type_levels), type = type_levels) |>
  readr::write_csv(file.path(out_dir_do, "type_index.csv"))

tibble::tibble(
  t0_year       = time_info$t0,
  t_scale_years = time_info$scale_years,
  K_u           = dat_do$K_u,
  coord_tol     = time_info$tol
) |>
  readr::write_csv(file.path(out_dir_do, "time_config.csv"))

# ---- Full parameter summary ----
summ_all_do <- fit_bb_tg_ctx_stz$summary()
readr::write_csv(as.data.frame(summ_all_do),
                 file.path(out_dir_do, "summary_all.csv"))

# ---- Core scalar parameters (no GP hyperparameters) ----
core_vars_do <- c("alpha", "beta_date", "gamma_ctx", "phi")
summ_core_do <- tryCatch(
  fit_bb_tg_ctx_stz$summary(variables = core_vars_do),
  error = function(e) tibble::tibble(note = conditionMessage(e))
)
readr::write_csv(as.data.frame(summ_core_do),
                 file.path(out_dir_do, "summary_core.csv"))

# ---- Type fixed effects (sum-to-zero; beta_type[1..T]) ----
type_draws_do <- tryCatch(
  posterior::as_draws_df(
    fit_bb_tg_ctx_stz$draws(variables = "beta_type")
  ),
  error = function(e) NULL
)

if (!is.null(type_draws_do)) {
  bt_cols_do <- grep("^beta_type\\[\\d+\\]$", names(type_draws_do), value = TRUE)
  idxs_do    <- as.integer(gsub(".*\\[(\\d+)\\]$", "\\1", bt_cols_do))
  bt_cols_do <- bt_cols_do[order(idxs_do)]

  summ_type_do <- tibble::tibble(
    variable = bt_cols_do,
    type_id  = seq_along(bt_cols_do),
    type     = type_levels[seq_along(bt_cols_do)],
    median   = sapply(bt_cols_do, \(nm) stats::median(type_draws_do[[nm]])),
    lo95     = sapply(bt_cols_do, \(nm) stats::quantile(type_draws_do[[nm]], 0.025)),
    hi95     = sapply(bt_cols_do, \(nm) stats::quantile(type_draws_do[[nm]], 0.975))
  )
  readr::write_csv(summ_type_do,
                   file.path(out_dir_do, "summary_beta_type_fixed.csv"))
} else {
  message("beta_type not found in draws; summary_beta_type_fixed.csv not written.")
}

# ---- theta_bar quantities ----
for (qty in c("theta_bar", "theta_bar_out", "theta_bar_in")) {
  if (has_var(fit_bb_tg_ctx_stz, qty)) {
    th_do <- fit_bb_tg_ctx_stz$summary(variables = qty) %>%
      dplyr::mutate(
        row = as.integer(gsub(paste0("^", qty, "\\[(\\d+)\\]$"), "\\1", variable))
      ) %>%
      dplyr::relocate(row, .before = 1)
    readr::write_csv(as.data.frame(th_do),
                     file.path(out_dir_do, paste0("summary_", qty, ".csv")))
  }
}

# ---- Diagnostics ----
diag_df_do <- tryCatch(
  as.data.frame(fit_bb_tg_ctx_stz$diagnostic_summary()),
  error = function(e) NULL
)
if (!is.null(diag_df_do))
  readr::write_csv(diag_df_do, file.path(out_dir_do, "diagnostic_summary.csv"))

# ---- Provenance ----
run_meta_do <- list(
  model_file    = stan_file_do,
  fit_name      = fit_name_do,
  results_dir   = normalizePath(out_dir_do),
  N             = dat_do$N,
  T             = dat_do$T,
  K_u           = dat_do$K_u,
  chains        = 4L,
  iter_warmup   = 2000L,
  iter_sampling = 2000L,
  adapt_delta   = 0.90,
  seed          = 2025L
)
readr::write_file(
  jsonlite::toJSON(run_meta_do, auto_unbox = TRUE, pretty = TRUE),
  file.path(out_dir_do, "run_metadata.json")
)

# ==========================================================
#   CONSOLE SUMMARY
# ==========================================================

message("\nKey scalar parameters:")
print(tibble::as_tibble(summ_core_do))

message("\nType fixed effects (sum-to-zero beta_type):")
if (exists("summ_type_do")) print(summ_type_do) else message("Not available.")
