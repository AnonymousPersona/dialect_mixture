# ==========================================================
#  02_run_bb_tgm_ctx_stgp_ctx_x_stgp_stz.R
#
#  Interaction model: adds a second spatiotemporal GP (g_site)
#  for the EIR contrast on top of Model 01's structure.
#  delta_site[s] = gamma_ctx + g_site[s] is the primary
#  spatial mapping quantity.
#
#  Requires: 00_setup.R (provides dat_stgp_ctx, sites_tbl,
#            type_levels, time_info, has_var)
# ==========================================================

source(here::here("r", "repo_models", "00_setup.R"))

# ----------------------------------------------------------
#  PATHS
# ----------------------------------------------------------

fit_name2 <- "02_bb_tgm_ctx_stgp_ctx_x_stgp_stz"
out_dir2  <- file.path(results_root, fit_name2)
dir.create(out_dir2, recursive = TRUE, showWarnings = FALSE)

csv_dir2 <- file.path(out_dir2, "csv")
dir.create(csv_dir2, recursive = TRUE, showWarnings = FALSE)

# ----------------------------------------------------------
#  COMPILE & SAMPLE
# ----------------------------------------------------------

stan_file2 <- file.path(
  here::here("stan"),
  paste0(fit_name2, ".stan")
)
if (!file.exists(path.expand(stan_file2))) stop("Stan file not found: ", stan_file2)
mod2 <- cmdstanr::cmdstan_model(stan_file2)

fit_bb_tgm_ctx_stgp_ctx_x_stgp_stz <- mod2$sample(
  data             = dat_stgp_ctx,
  output_dir       = csv_dir2,
  seed             = 2025,
  chains           = 4, parallel_chains = 4,
  iter_warmup      = 2000, iter_sampling  = 2000,
  adapt_delta      = 0.95, max_treedepth  = 12,
  refresh          = 50
)

# ----------------------------------------------------------
#  LOO
# ----------------------------------------------------------

if (has_var(fit_bb_tgm_ctx_stgp_ctx_x_stgp_stz, "log_lik")) {
  ll_mat2 <- fit_bb_tgm_ctx_stgp_ctx_x_stgp_stz$draws("log_lik", format = "matrix")

  if (is.matrix(ll_mat2) && ncol(ll_mat2) > 20000) {
    set.seed(1); keep2 <- sort(sample.int(ncol(ll_mat2), 20000))
    ll_mat2 <- ll_mat2[, keep2, drop = FALSE]
  }

  if (is.matrix(ll_mat2) && ncol(ll_mat2) > 0) {
    loo_obj2 <- loo::loo(ll_mat2, save_psis = TRUE, moment_match = FALSE)

    est2 <- loo_obj2$estimates
    writeLines(paste0(
      "elpd_loo = ", sprintf("%.3f", est2["elpd_loo", "Estimate"]),
      " (SE = ", sprintf("%.3f", est2["elpd_loo", "SE"]), ")\n",
      "p_loo    = ", sprintf("%.3f", est2["p_loo",    "Estimate"]), "\n",
      "looic    = ", sprintf("%.3f", est2["looic",     "Estimate"]), "\n",
      "---------\n",
      "Pareto k diagnostics:\n",
      paste(capture.output(print(loo::pareto_k_table(loo_obj2))), collapse = "\n"), "\n"
    ), file.path(out_dir2, "loo.txt"))

    readr::write_csv(
      tibble::tibble(
        metric   = rownames(est2),
        estimate = as.numeric(est2[, "Estimate"]),
        se       = as.numeric(est2[, "SE"])
      ),
      file.path(out_dir2, "loo_estimates.csv")
    )
    pk2 <- loo::pareto_k_table(loo_obj2)
    suppressWarnings(
      readr::write_csv(as.data.frame(pk2), file.path(out_dir2, "pareto_k.csv"))
    )
    saveRDS(loo_obj2, file.path(out_dir2, "loo.rds"))
  } else {
    readr::write_lines("Found 'log_lik' but it contained 0 columns.",
                       file.path(out_dir2, "loo.txt"))
  }
} else {
  readr::write_lines("log_lik not found in fitted draws; LOO not computed.",
                     file.path(out_dir2, "loo.txt"))
}

# ----------------------------------------------------------
#  SAVE FIT OBJECT
# ----------------------------------------------------------

saveRDS(fit_bb_tgm_ctx_stgp_ctx_x_stgp_stz,
        file.path(out_dir2, paste0(fit_name2, ".rds")))

# ----------------------------------------------------------
#  INDEX TABLES
# ----------------------------------------------------------

tibble::tibble(type_id = seq_along(type_levels), type = type_levels) |>
  readr::write_csv(file.path(out_dir2, "type_index.csv"))

sites_tbl |>
  dplyr::mutate(site_id = dplyr::row_number()) |>
  readr::write_csv(file.path(out_dir2, "site_index.csv"))

tibble::tibble(
  t0_year       = time_info$t0,
  t_scale_years = time_info$scale_years,
  K_u           = dat_stgp_ctx$K_u,
  coord_tol     = time_info$tol
) |>
  readr::write_csv(file.path(out_dir2, "time_config.csv"))

# ----------------------------------------------------------
#  PARAMETER SUMMARIES
# ----------------------------------------------------------

summ_all2 <- fit_bb_tgm_ctx_stgp_ctx_x_stgp_stz$summary()
readr::write_csv(summ_all2, file.path(out_dir2, "summary_all.csv"))

core_vars2 <- c("alpha", "beta_date", "gamma_ctx", "phi",
                "sigma_gp", "rho_s", "rho_t", "sigma_ctx")
summ_core2 <- tryCatch(
  fit_bb_tgm_ctx_stgp_ctx_x_stgp_stz$summary(variables = core_vars2),
  error = function(e) tibble::tibble(note = conditionMessage(e))
)
readr::write_csv(summ_core2, file.path(out_dir2, "summary_core.csv"))

# ---- Type fixed effects ----
type_draws2 <- tryCatch(
  posterior::as_draws_df(
    fit_bb_tgm_ctx_stgp_ctx_x_stgp_stz$draws(variables = "beta_type")
  ),
  error = function(e) NULL
)
if (!is.null(type_draws2)) {
  bt_cols2 <- grep("^beta_type\\[\\d+\\]$", names(type_draws2), value = TRUE)
  bt_cols2 <- bt_cols2[order(as.integer(gsub(".*\\[(\\d+)\\]$", "\\1", bt_cols2)))]
  summ_type2 <- tibble::tibble(
    variable = bt_cols2,
    type_id  = seq_along(bt_cols2),
    type     = type_levels[seq_along(bt_cols2)],
    median   = sapply(bt_cols2, \(nm) stats::median(type_draws2[[nm]])),
    lo95     = sapply(bt_cols2, \(nm) stats::quantile(type_draws2[[nm]], 0.025)),
    hi95     = sapply(bt_cols2, \(nm) stats::quantile(type_draws2[[nm]], 0.975))
  )
  readr::write_csv(summ_type2, file.path(out_dir2, "summary_beta_type_fixed.csv"))
} else {
  message("beta_type not found in draws; summary_beta_type_fixed.csv not written.")
}

# ---- f_site: baseline dialect GP surface ----
if (has_var(fit_bb_tgm_ctx_stgp_ctx_x_stgp_stz, "f_site")) {
  fit_bb_tgm_ctx_stgp_ctx_x_stgp_stz$summary(variables = "f_site") |>
    dplyr::mutate(
      site_id = as.integer(gsub("^f_site\\[(\\d+)\\]$", "\\1", variable))
    ) |>
    dplyr::left_join(
      sites_tbl |> dplyr::mutate(site_id = dplyr::row_number()),
      by = "site_id"
    ) |>
    dplyr::relocate(site_id, latitude_site, longitude_site, .after = variable) |>
    readr::write_csv(file.path(out_dir2, "summary_f_site.csv"))
}

# ---- g_site: residual EIR-contrast GP surface ----
if (has_var(fit_bb_tgm_ctx_stgp_ctx_x_stgp_stz, "g_site")) {
  fit_bb_tgm_ctx_stgp_ctx_x_stgp_stz$summary(variables = "g_site") |>
    dplyr::mutate(
      site_id = as.integer(gsub("^g_site\\[(\\d+)\\]$", "\\1", variable))
    ) |>
    dplyr::left_join(
      sites_tbl |> dplyr::mutate(site_id = dplyr::row_number()),
      by = "site_id"
    ) |>
    dplyr::relocate(site_id, latitude_site, longitude_site, .after = variable) |>
    readr::write_csv(file.path(out_dir2, "summary_g_site.csv"))
}

# ---- delta_site: total EIR shift per site ----
if (has_var(fit_bb_tgm_ctx_stgp_ctx_x_stgp_stz, "delta_site")) {
  summ_deltasite2 <-
    fit_bb_tgm_ctx_stgp_ctx_x_stgp_stz$summary(variables = "delta_site") |>
    dplyr::mutate(
      site_id = as.integer(gsub("^delta_site\\[(\\d+)\\]$", "\\1", variable))
    ) |>
    dplyr::left_join(
      sites_tbl |> dplyr::mutate(site_id = dplyr::row_number()),
      by = "site_id"
    ) |>
    dplyr::relocate(site_id, latitude_site, longitude_site, .after = variable)
  readr::write_csv(summ_deltasite2, file.path(out_dir2, "summary_delta_site.csv"))
}

# ---- theta_bar quantities ----
for (qty in c("theta_bar", "theta_bar_out", "theta_bar_in")) {
  if (has_var(fit_bb_tgm_ctx_stgp_ctx_x_stgp_stz, qty)) {
    fit_bb_tgm_ctx_stgp_ctx_x_stgp_stz$summary(variables = qty) |>
      dplyr::mutate(
        row = as.integer(gsub(paste0("^", qty, "\\[(\\d+)\\]$"), "\\1", variable))
      ) |>
      dplyr::relocate(row, .before = 1) |>
      readr::write_csv(file.path(out_dir2, paste0("summary_", qty, ".csv")))
  }
}

# ----------------------------------------------------------
#  DIAGNOSTICS
# ----------------------------------------------------------

diag_df2 <- tryCatch(
  as.data.frame(fit_bb_tgm_ctx_stgp_ctx_x_stgp_stz$diagnostic_summary()),
  error = function(e) NULL
)
if (!is.null(diag_df2))
  readr::write_csv(diag_df2, file.path(out_dir2, "diagnostic_summary.csv"))

# ----------------------------------------------------------
#  PROVENANCE
# ----------------------------------------------------------

readr::write_file(
  jsonlite::toJSON(list(
    model_file    = stan_file2,
    fit_name      = fit_name2,
    results_dir   = normalizePath(out_dir2),
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
  file.path(out_dir2, "run_metadata.json")
)

# ----------------------------------------------------------
#  CONSOLE SUMMARY
# ----------------------------------------------------------

message("\nKey scalar parameters (incl. sigma_ctx):")
print(tibble::as_tibble(summ_core2))

message("\nType fixed effects (sum-to-zero beta_type):")
if (exists("summ_type2")) print(summ_type2) else message("Not available.")

message("\nGP hyperparameters:")
print(tibble::as_tibble(
  fit_bb_tgm_ctx_stgp_ctx_x_stgp_stz$summary(
    variables = c("sigma_gp", "sigma_ctx", "rho_s", "rho_t")
  )
))

message("\ndelta_site (total EIR shift; first 10 sites):")
if (exists("summ_deltasite2")) print(head(summ_deltasite2, 10)) else message("Not available.")
