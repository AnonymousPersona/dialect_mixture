# ==========================================================
#  10_run_bb_tg_ctx_stgp_slope_stz.R
#
#  Site × date-trend interaction model: adds a spatial GP for
#  the slope deviation b_site[S] on top of Model 01's
#  structure.  Heavier to sample: iter raised to 3000/4000,
#  adapt_delta = 0.99, max_treedepth = 14.
#
#  Requires: 00_setup.R (provides dat_stgp_ctx, sites_tbl,
#            type_levels, time_info, has_var)
# ==========================================================

source(here::here("r", "repo_models", "00_setup.R"))

# ==========================================================
#       bb_tg_ctx_stgp_slope_stz  —  run + save
#
#  Site × date-trend interaction model.
#  Adds a spatial GP for the slope deviation b_site[S] on top
#  of Model 1's full structure.  The data block is identical
#  to dat_stgp_ctx; no new data builder is needed.
# ==========================================================

stopifnot(
  exists("dat_stgp_ctx"),
  exists("has_var"),
  exists("type_levels"),
  exists("sites_tbl"),
  exists("time_info")
)

# ---------- paths ----------
fit_name_sl     <- "10_bb_tg_ctx_stgp_slope_stz"
out_dir_sl      <- file.path(results_root, fit_name_sl)
dir.create(out_dir_sl, recursive = TRUE, showWarnings = FALSE)

csv_dir_sl <- file.path(out_dir_sl, "csv")
dir.create(csv_dir_sl, recursive = TRUE, showWarnings = FALSE)

# ==========================================================
#   COMPILE & SAMPLE
#   dat_stgp_ctx has exactly the fields this model's data
#   block declares; no subsetting or augmentation needed.
# ==========================================================

stan_file_sl <- file.path(
  here::here("stan"),
  paste0(fit_name_sl, ".stan")
)
if (!file.exists(path.expand(stan_file_sl))) stop("Stan file not found: ", stan_file_sl)
mod_sl <- cmdstanr::cmdstan_model(stan_file_sl)

# adapt_delta and max_treedepth raised relative to Model 1:
# two independent GP Cholesky factorisations (f_site and b_site)
# plus the additional (sigma_slope, rho_slope) funnel geometry
# make this model appreciably harder to sample.
fit_bb_tg_ctx_stgp_slope_stz <- mod_sl$sample(
  data             = dat_stgp_ctx,
  output_dir       = csv_dir_sl,
  seed             = 2025,
  chains           = 4, parallel_chains = 4,
  iter_warmup      = 3000, iter_sampling  = 4000,   # ← increased from 2000/2000
  adapt_delta      = 0.99,                          # ← raised from 0.97
  max_treedepth    = 14,
  refresh          = 100
)

# ==========================================================
#   SAVE: summaries, LOO, indices, metadata
# ==========================================================

# ---- LOO ----
if (has_var(fit_bb_tg_ctx_stgp_slope_stz, "log_lik")) {
  ll_mat_sl <- fit_bb_tg_ctx_stgp_slope_stz$draws("log_lik", format = "matrix")

  if (is.matrix(ll_mat_sl) && ncol(ll_mat_sl) > 20000) {
    set.seed(1); keep_sl <- sort(sample.int(ncol(ll_mat_sl), 20000))
    ll_mat_sl <- ll_mat_sl[, keep_sl, drop = FALSE]
  }

  if (is.matrix(ll_mat_sl) && ncol(ll_mat_sl) > 0) {
    loo_obj_sl <- loo::loo(ll_mat_sl, save_psis = TRUE, moment_match = FALSE)

    est_sl <- loo_obj_sl$estimates
    txt_sl <- paste0(
      "elpd_loo = ", sprintf("%.3f", est_sl["elpd_loo", "Estimate"]),
      " (SE = ", sprintf("%.3f", est_sl["elpd_loo", "SE"]), ")\n",
      "p_loo    = ", sprintf("%.3f", est_sl["p_loo",    "Estimate"]), "\n",
      "looic    = ", sprintf("%.3f", est_sl["looic",     "Estimate"]), "\n",
      "---------\n",
      "Pareto k diagnostics:\n",
      paste(capture.output(print(loo::pareto_k_table(loo_obj_sl))), collapse = "\n"), "\n"
    )
    writeLines(txt_sl, file.path(out_dir_sl, "loo.txt"))

    readr::write_csv(
      tibble::tibble(
        metric   = rownames(est_sl),
        estimate = as.numeric(est_sl[, "Estimate"]),
        se       = as.numeric(est_sl[, "SE"])
      ),
      file.path(out_dir_sl, "loo_estimates.csv")
    )

    pk_sl <- loo::pareto_k_table(loo_obj_sl)
    suppressWarnings(
      readr::write_csv(as.data.frame(pk_sl), file.path(out_dir_sl, "pareto_k.csv"))
    )

    saveRDS(loo_obj_sl, file.path(out_dir_sl, "loo.rds"))
    rm(ll_mat_sl); gc()
  } else {
    readr::write_lines(
      "Found 'log_lik' but it contained 0 columns.",
      file.path(out_dir_sl, "loo.txt")
    )
  }
} else {
  readr::write_lines(
    "log_lik not found in fitted draws; LOO not computed.",
    file.path(out_dir_sl, "loo.txt")
  )
}

# ---- Save fit object ----
saveRDS(fit_bb_tg_ctx_stgp_slope_stz,
        file.path(out_dir_sl, paste0(fit_name_sl, ".rds")))

# ---- Index tables ----
tibble::tibble(type_id = seq_along(type_levels), type = type_levels) |>
  readr::write_csv(file.path(out_dir_sl, "type_index.csv"))

sites_tbl |>
  dplyr::mutate(site_id = dplyr::row_number()) |>
  readr::write_csv(file.path(out_dir_sl, "site_index.csv"))

tibble::tibble(
  t0_year       = time_info$t0,
  t_scale_years = time_info$scale_years,
  K_u           = dat_stgp_ctx$K_u,
  coord_tol     = time_info$tol
) |>
  readr::write_csv(file.path(out_dir_sl, "time_config.csv"))

# ---- Full parameter summary (exclude K_site / K_slope if present) ----
# Both are local temporaries in Stan and should not appear in draws,
# but the guard is retained as a precaution.
{
  md_vars_sl   <- fit_bb_tg_ctx_stgp_slope_stz$metadata()$stan_variables
  vars_to_summ <- md_vars_sl[!grepl("^K_site$|^K_slope$", md_vars_sl)]
  summ_all_sl  <- fit_bb_tg_ctx_stgp_slope_stz$summary(variables = vars_to_summ)
  readr::write_csv(as.data.frame(summ_all_sl),
                   file.path(out_dir_sl, "summary_all.csv"))
  rm(summ_all_sl); gc()
}

# ---- Core scalar parameters (adds slope GP hyperparameters) ----
core_vars_sl <- c("alpha", "beta_date", "gamma_ctx", "phi",
                  "sigma_gp", "rho_s", "rho_t",
                  "sigma_slope", "rho_slope")
summ_core_sl <- tryCatch(
  fit_bb_tg_ctx_stgp_slope_stz$summary(variables = core_vars_sl),
  error = function(e) tibble::tibble(note = conditionMessage(e))
)
readr::write_csv(as.data.frame(summ_core_sl),
                 file.path(out_dir_sl, "summary_core.csv"))

# ---- Type fixed effects (sum-to-zero; beta_type[1..T]) ----
type_draws_sl <- tryCatch(
  posterior::as_draws_df(
    fit_bb_tg_ctx_stgp_slope_stz$draws(variables = "beta_type")
  ),
  error = function(e) NULL
)

if (!is.null(type_draws_sl)) {
  bt_cols_sl <- grep("^beta_type\\[\\d+\\]$", names(type_draws_sl), value = TRUE)
  idxs_sl    <- as.integer(gsub(".*\\[(\\d+)\\]$", "\\1", bt_cols_sl))
  bt_cols_sl <- bt_cols_sl[order(idxs_sl)]

  summ_type_sl <- tibble::tibble(
    variable = bt_cols_sl,
    type_id  = seq_along(bt_cols_sl),
    type     = type_levels[seq_along(bt_cols_sl)],
    median   = sapply(bt_cols_sl, \(nm) stats::median(type_draws_sl[[nm]])),
    lo95     = sapply(bt_cols_sl, \(nm) stats::quantile(type_draws_sl[[nm]], 0.025)),
    hi95     = sapply(bt_cols_sl, \(nm) stats::quantile(type_draws_sl[[nm]], 0.975))
  )
  readr::write_csv(summ_type_sl,
                   file.path(out_dir_sl, "summary_beta_type_fixed.csv"))
  rm(type_draws_sl); gc()
} else {
  message("beta_type not found in draws; summary_beta_type_fixed.csv not written.")
}

# ---- GP intercept surface: f_site[1..S] ----
if (has_var(fit_bb_tg_ctx_stgp_slope_stz, "f_site")) {
  summ_fsite_sl <- fit_bb_tg_ctx_stgp_slope_stz$summary(variables = "f_site") %>%
    dplyr::mutate(
      site_id = as.integer(gsub("^f_site\\[(\\d+)\\]$", "\\1", variable))
    ) %>%
    dplyr::left_join(
      sites_tbl %>% dplyr::mutate(site_id = dplyr::row_number()),
      by = "site_id"
    ) %>%
    dplyr::relocate(site_id, latitude_site, longitude_site, .after = variable)
  readr::write_csv(as.data.frame(summ_fsite_sl),
                   file.path(out_dir_sl, "summary_f_site.csv"))
}

# ---- GP slope deviation surface: b_site[1..S] ----
# Signed deviation of each site's temporal trend from the global beta_date.
# Positive → faster koine adoption; negative → slower or resistant.
if (has_var(fit_bb_tg_ctx_stgp_slope_stz, "b_site")) {
  summ_bsite_sl <- fit_bb_tg_ctx_stgp_slope_stz$summary(variables = "b_site") %>%
    dplyr::mutate(
      site_id = as.integer(gsub("^b_site\\[(\\d+)\\]$", "\\1", variable))
    ) %>%
    dplyr::left_join(
      sites_tbl %>% dplyr::mutate(site_id = dplyr::row_number()),
      by = "site_id"
    ) %>%
    dplyr::relocate(site_id, latitude_site, longitude_site, .after = variable)
  readr::write_csv(as.data.frame(summ_bsite_sl),
                   file.path(out_dir_sl, "summary_b_site.csv"))
}

# ---- Site-level effective temporal trend: beta_date_site[1..S] ----
# Primary mapping quantity: beta_date + b_site[s].
# Comparable to the global beta_date from Model 1 but now resolved
# spatially.  Single-inscription sites will be near the global mean
# with wide posterior intervals (prior dominates).
if (has_var(fit_bb_tg_ctx_stgp_slope_stz, "beta_date_site")) {
  summ_bdsite_sl <- fit_bb_tg_ctx_stgp_slope_stz$summary(
    variables = "beta_date_site"
  ) %>%
    dplyr::mutate(
      site_id = as.integer(
        gsub("^beta_date_site\\[(\\d+)\\]$", "\\1", variable)
      )
    ) %>%
    dplyr::left_join(
      sites_tbl %>% dplyr::mutate(site_id = dplyr::row_number()),
      by = "site_id"
    ) %>%
    dplyr::relocate(site_id, latitude_site, longitude_site, .after = variable)
  readr::write_csv(as.data.frame(summ_bdsite_sl),
                   file.path(out_dir_sl, "summary_beta_date_site.csv"))
}

# ---- theta_bar quantities ----
for (qty in c("theta_bar", "theta_bar_out", "theta_bar_in")) {
  if (has_var(fit_bb_tg_ctx_stgp_slope_stz, qty)) {
    th_sl <- fit_bb_tg_ctx_stgp_slope_stz$summary(variables = qty) %>%
      dplyr::mutate(
        row = as.integer(gsub(paste0("^", qty, "\\[(\\d+)\\]$"), "\\1", variable))
      ) %>%
      dplyr::relocate(row, .before = 1)
    readr::write_csv(as.data.frame(th_sl),
                     file.path(out_dir_sl, paste0("summary_", qty, ".csv")))
  }
}

# ---- Diagnostics ----
diag_df_sl <- tryCatch(
  as.data.frame(fit_bb_tg_ctx_stgp_slope_stz$diagnostic_summary()),
  error = function(e) NULL
)
if (!is.null(diag_df_sl))
  readr::write_csv(diag_df_sl, file.path(out_dir_sl, "diagnostic_summary.csv"))

# ---- Provenance ----
run_meta_sl <- list(
  model_file    = stan_file_sl,
  fit_name      = fit_name_sl,
  results_dir   = normalizePath(out_dir_sl),
  N             = dat_stgp_ctx$N,
  S             = dat_stgp_ctx$S,
  T             = dat_stgp_ctx$T,
  K_u           = dat_stgp_ctx$K_u,
  chains        = 4L,
  iter_warmup   = 2000L,
  iter_sampling = 2000L,
  adapt_delta   = 0.97,
  seed          = 2025L
)
readr::write_file(
  jsonlite::toJSON(run_meta_sl, auto_unbox = TRUE, pretty = TRUE),
  file.path(out_dir_sl, "run_metadata.json")
)

# ==========================================================
#   CONSOLE SUMMARY
# ==========================================================

message("\nCore scalar parameters (incl. slope GP hyperparameters):")
print(tibble::as_tibble(summ_core_sl))

message("\nType fixed effects (sum-to-zero beta_type):")
if (exists("summ_type_sl")) print(summ_type_sl) else message("Not available.")

message("\nIntercept GP hyperparameters:")
print(tibble::as_tibble(
  fit_bb_tg_ctx_stgp_slope_stz$summary(variables = c("sigma_gp", "rho_s", "rho_t"))
))

message("\nSlope GP hyperparameters:")
print(tibble::as_tibble(
  fit_bb_tg_ctx_stgp_slope_stz$summary(variables = c("sigma_slope", "rho_slope"))
))

message("\nSite-level effective temporal trend (beta_date_site, first 10 sites):")
if (exists("summ_bdsite_sl")) print(head(tibble::as_tibble(summ_bdsite_sl), 10)) else message("Not available.")
