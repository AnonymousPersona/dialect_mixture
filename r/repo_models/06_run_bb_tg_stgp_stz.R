# ==========================================================
#  06_run_bb_tg_stgp_stz.R
#
#  No-context STGP model: retains f_site GP, beta_date,
#  beta_type, phi; removes gamma_ctx entirely.  Pools EIR
#  and non-EIR tokens within each inscription.
#
#  Requires: 00_setup.R (provides dat_stgp_ctx, sites_tbl,
#            type_levels, time_info, has_var)
# ==========================================================

source(here::here("r", "repo_models", "00_setup.R"))

# ==========================================================
#       bb_tg_stgp_stz  —  run + save
#
#  No-context STGP model: keeps f_site GP, beta_date,
#  beta_type; removes gamma_ctx entirely; pools EIR and
#  non-EIR tokens within each inscription.
#  dat_stgp_ctx is passed directly — no subsetting needed
#  since the Stan data block retains full site structure.
# ==========================================================

stopifnot(
  exists("dat_stgp_ctx"),
  exists("has_var"),
  exists("type_levels"),
  exists("sites_tbl"),
  exists("time_info")
)

# ---------- paths ----------
fit_name_nc     <- "06_bb_tg_stgp_stz"
out_dir_nc      <- file.path(results_root, fit_name_nc)
dir.create(out_dir_nc, recursive = TRUE, showWarnings = FALSE)

csv_dir_nc <- file.path(out_dir_nc, "csv")
dir.create(csv_dir_nc, recursive = TRUE, showWarnings = FALSE)

# ==========================================================
#   COMPILE & SAMPLE
#   dat_stgp_ctx passes directly: the Stan data block accepts
#   y1/n1/y2/n2 separately and pools them in transformed data.
# ==========================================================

stan_file_nc <- file.path(
  here::here("stan"),
  paste0(fit_name_nc, ".stan")
)
if (!file.exists(path.expand(stan_file_nc))) stop("Stan file not found: ", stan_file_nc)
mod_nc <- cmdstanr::cmdstan_model(stan_file_nc)

fit_bb_tg_stgp_stz <- mod_nc$sample(
  data             = dat_stgp_ctx,
  output_dir       = csv_dir_nc,
  seed             = 2025,
  chains           = 4, parallel_chains = 4,
  iter_warmup      = 2000, iter_sampling  = 2000,
  adapt_delta      = 0.95, max_treedepth  = 12,
  refresh          = 50
)

# ==========================================================
#   SAVE: summaries, LOO, indices, metadata
# ==========================================================

# ---- LOO ----
if (has_var(fit_bb_tg_stgp_stz, "log_lik")) {
  ll_mat_nc <- fit_bb_tg_stgp_stz$draws("log_lik", format = "matrix")

  if (is.matrix(ll_mat_nc) && ncol(ll_mat_nc) > 20000) {
    set.seed(1); keep_nc <- sort(sample.int(ncol(ll_mat_nc), 20000))
    ll_mat_nc <- ll_mat_nc[, keep_nc, drop = FALSE]
  }

  if (is.matrix(ll_mat_nc) && ncol(ll_mat_nc) > 0) {
    loo_obj_nc <- loo::loo(ll_mat_nc, save_psis = TRUE, moment_match = FALSE)

    est_nc <- loo_obj_nc$estimates
    txt_nc <- paste0(
      "elpd_loo = ", sprintf("%.3f", est_nc["elpd_loo", "Estimate"]),
      " (SE = ", sprintf("%.3f", est_nc["elpd_loo", "SE"]), ")\n",
      "p_loo    = ", sprintf("%.3f", est_nc["p_loo",    "Estimate"]), "\n",
      "looic    = ", sprintf("%.3f", est_nc["looic",     "Estimate"]), "\n",
      "---------\n",
      "Pareto k diagnostics:\n",
      paste(capture.output(print(loo::pareto_k_table(loo_obj_nc))), collapse = "\n"), "\n"
    )
    writeLines(txt_nc, file.path(out_dir_nc, "loo.txt"))

    readr::write_csv(
      tibble::tibble(
        metric   = rownames(est_nc),
        estimate = as.numeric(est_nc[, "Estimate"]),
        se       = as.numeric(est_nc[, "SE"])
      ),
      file.path(out_dir_nc, "loo_estimates.csv")
    )

    pk_nc <- loo::pareto_k_table(loo_obj_nc)
    suppressWarnings(
      readr::write_csv(as.data.frame(pk_nc), file.path(out_dir_nc, "pareto_k.csv"))
    )

    saveRDS(loo_obj_nc, file.path(out_dir_nc, "loo.rds"))
    rm(ll_mat_nc); gc()
  } else {
    readr::write_lines(
      "Found 'log_lik' but it contained 0 columns.",
      file.path(out_dir_nc, "loo.txt")
    )
  }
} else {
  readr::write_lines(
    "log_lik not found in fitted draws; LOO not computed.",
    file.path(out_dir_nc, "loo.txt")
  )
}

# ---- Save fit object ----
saveRDS(fit_bb_tg_stgp_stz,
        file.path(out_dir_nc, paste0(fit_name_nc, ".rds")))

# ---- Index tables ----
tibble::tibble(type_id = seq_along(type_levels), type = type_levels) |>
  readr::write_csv(file.path(out_dir_nc, "type_index.csv"))

sites_tbl |>
  dplyr::mutate(site_id = dplyr::row_number()) |>
  readr::write_csv(file.path(out_dir_nc, "site_index.csv"))

tibble::tibble(
  t0_year       = time_info$t0,
  t_scale_years = time_info$scale_years,
  K_u           = dat_stgp_ctx$K_u,
  coord_tol     = time_info$tol
) |>
  readr::write_csv(file.path(out_dir_nc, "time_config.csv"))

# ---- Full parameter summary (exclude K_site if present) ----
{
  md_vars_nc   <- fit_bb_tg_stgp_stz$metadata()$stan_variables
  vars_to_summ <- md_vars_nc[!grepl("^K_site$", md_vars_nc)]
  summ_all_nc  <- fit_bb_tg_stgp_stz$summary(variables = vars_to_summ)
  readr::write_csv(as.data.frame(summ_all_nc),
                   file.path(out_dir_nc, "summary_all.csv"))
  rm(summ_all_nc); gc()
}

# ---- Core scalar parameters (no gamma_ctx; GP hyperparameters included) ----
core_vars_nc <- c("alpha", "beta_date", "phi", "sigma_gp", "rho_s", "rho_t")
summ_core_nc <- tryCatch(
  fit_bb_tg_stgp_stz$summary(variables = core_vars_nc),
  error = function(e) tibble::tibble(note = conditionMessage(e))
)
readr::write_csv(as.data.frame(summ_core_nc),
                 file.path(out_dir_nc, "summary_core.csv"))

# ---- Type fixed effects (sum-to-zero; beta_type[1..T]) ----
type_draws_nc <- tryCatch(
  posterior::as_draws_df(
    fit_bb_tg_stgp_stz$draws(variables = "beta_type")
  ),
  error = function(e) NULL
)

if (!is.null(type_draws_nc)) {
  bt_cols_nc <- grep("^beta_type\\[\\d+\\]$", names(type_draws_nc), value = TRUE)
  idxs_nc    <- as.integer(gsub(".*\\[(\\d+)\\]$", "\\1", bt_cols_nc))
  bt_cols_nc <- bt_cols_nc[order(idxs_nc)]

  summ_type_nc <- tibble::tibble(
    variable = bt_cols_nc,
    type_id  = seq_along(bt_cols_nc),
    type     = type_levels[seq_along(bt_cols_nc)],
    median   = sapply(bt_cols_nc, \(nm) stats::median(type_draws_nc[[nm]])),
    lo95     = sapply(bt_cols_nc, \(nm) stats::quantile(type_draws_nc[[nm]], 0.025)),
    hi95     = sapply(bt_cols_nc, \(nm) stats::quantile(type_draws_nc[[nm]], 0.975))
  )
  readr::write_csv(summ_type_nc,
                   file.path(out_dir_nc, "summary_beta_type_fixed.csv"))
  rm(type_draws_nc); gc()
} else {
  message("beta_type not found in draws; summary_beta_type_fixed.csv not written.")
}

# ---- GP intercept surface: f_site[1..S] ----
# Primary output of the no-context model: the pure dialect surface
# unconfounded by EIR, directly comparable to f_site in Model 1.
if (has_var(fit_bb_tg_stgp_stz, "f_site")) {
  summ_fsite_nc <- fit_bb_tg_stgp_stz$summary(variables = "f_site") %>%
    dplyr::mutate(
      site_id = as.integer(gsub("^f_site\\[(\\d+)\\]$", "\\1", variable))
    ) %>%
    dplyr::left_join(
      sites_tbl %>% dplyr::mutate(site_id = dplyr::row_number()),
      by = "site_id"
    ) %>%
    dplyr::relocate(site_id, latitude_site, longitude_site, .after = variable)
  readr::write_csv(as.data.frame(summ_fsite_nc),
                   file.path(out_dir_nc, "summary_f_site.csv"))
}

# ---- theta_bar (pooled; no out/in split in the no-context model) ----
if (has_var(fit_bb_tg_stgp_stz, "theta_bar")) {
  th_nc <- fit_bb_tg_stgp_stz$summary(variables = "theta_bar") %>%
    dplyr::mutate(
      row = as.integer(gsub("^theta_bar\\[(\\d+)\\]$", "\\1", variable))
    ) %>%
    dplyr::relocate(row, .before = 1)
  readr::write_csv(as.data.frame(th_nc),
                   file.path(out_dir_nc, "summary_theta_bar.csv"))
}

# ---- Diagnostics ----
diag_df_nc <- tryCatch(
  as.data.frame(fit_bb_tg_stgp_stz$diagnostic_summary()),
  error = function(e) NULL
)
if (!is.null(diag_df_nc))
  readr::write_csv(diag_df_nc, file.path(out_dir_nc, "diagnostic_summary.csv"))

# ---- Provenance ----
run_meta_nc <- list(
  model_file    = stan_file_nc,
  fit_name      = fit_name_nc,
  results_dir   = normalizePath(out_dir_nc),
  N             = dat_stgp_ctx$N,
  S             = dat_stgp_ctx$S,
  T             = dat_stgp_ctx$T,
  K_u           = dat_stgp_ctx$K_u,
  chains        = 4L,
  iter_warmup   = 2000L,
  iter_sampling = 2000L,
  adapt_delta   = 0.95,
  seed          = 2025L
)
readr::write_file(
  jsonlite::toJSON(run_meta_nc, auto_unbox = TRUE, pretty = TRUE),
  file.path(out_dir_nc, "run_metadata.json")
)

# ==========================================================
#   CONSOLE SUMMARY
# ==========================================================

message("\nCore scalar parameters (no gamma_ctx):")
print(tibble::as_tibble(summ_core_nc))

message("\nType fixed effects (sum-to-zero beta_type):")
if (exists("summ_type_nc")) print(summ_type_nc) else message("Not available.")

message("\nGP hyperparameters:")
print(tibble::as_tibble(
  fit_bb_tg_stgp_stz$summary(variables = c("sigma_gp", "rho_s", "rho_t"))
))
