# ==========================================================
#  09_run_bb_tg_ctx_stgp_ctx_dial_stz.R
#
#  EIR-contrast dialect interaction model: keeps Model 01's
#  STGP structure and adds dialect-group × EIR fixed effects
#  (delta_ctx[D], sum-to-zero) as the parametric counterpart
#  to Model 02's g_site GP.
#
#  Requires: 00_setup.R (provides dat_stgp_ctx, agg_data_dialect,
#            sites_tbl, type_levels, time_info, has_var,
#            ensure_std_time, t0_year, t_scale_years)
# ==========================================================

source(here::here("r", "repo_models", "00_setup.R"))

# ==========================================================
#       bb_tg_ctx_stgp_ctx_dial_stz  —  run + save
#
#  EIR-contrast dialect interaction model.
#  Keeps Model 1's full STGP structure and adds dialect-group
#  × EIR interaction fixed effects (delta_ctx[D], sum-to-zero)
#  as the parametric counterpart to Model 2's g_site GP.
#  Data = dat_stgp_ctx augmented with D and dialect[N].
# ==========================================================

stopifnot(
  exists("dat_stgp_ctx"),
  exists("has_var"),
  exists("type_levels"),
  exists("sites_tbl"),
  exists("time_info"),
  exists("ensure_std_time"),
  exists("t0_year"),
  exists("t_scale_years")
)

# ---------- paths ----------
fit_name_cd     <- "09_bb_tg_ctx_stgp_ctx_dial_stz"
out_dir_cd      <- file.path(results_root, fit_name_cd)
dir.create(out_dir_cd, recursive = TRUE, showWarnings = FALSE)

csv_dir_cd <- file.path(out_dir_cd, "csv")
dir.create(csv_dir_cd, recursive = TRUE, showWarnings = FALSE)

# ==========================================================
#   DATA BUILDER
#   Augments dat_stgp_ctx with dialect-group EIR interaction
#   inputs (D, dialect[N]).  Dialect recoding is identical to
#   build_data_dialect: four fixed levels in alphabetical order
#   (attic=1, doric=2, ionic=3, other=4); any dialect_group
#   value not in {attic, doric, ionic} — including NA — is
#   assigned to "other".
#
#   Row alignment: get_agg_data(require_dialect = TRUE) passes
#   through make_agg_data with the same exclude_types and no
#   row filtering, so its inscription order is identical to
#   dat_stgp_ctx.  The N-match stopifnot below enforces this.
# ==========================================================

build_dialect_augment <- function(dat_base, t0, scale_years) {
  # Fetch dialect-aware aggregated data
  df_d <- get_agg_data(require_dialect = TRUE)

  # Time-standardise with the same parameters as dat_base so that
  # rows correspond (standardisation does not change row order).
  df_d <- ensure_std_time(df_d, t0 = t0, scale_years = scale_years)

  # Row count must match exactly
  if (nrow(df_d) != dat_base$N) {
    stop(sprintf(
      "Row mismatch: dialect data has %d rows, dat_stgp_ctx has %d. ",
      nrow(df_d), dat_base$N
    ))
  }

  # Dialect-group recoding: four fixed alphabetical levels
  dialect_levels <- c("attic", "doric", "ionic", "other")
  df_d <- df_d %>%
    dplyr::mutate(
      dialect_4 = dplyr::case_when(
        is.na(dialect_group)     ~ "other",
        dialect_group == "attic" ~ "attic",
        dialect_group == "doric" ~ "doric",
        dialect_group == "ionic" ~ "ionic",
        TRUE                     ~ "other"
      ),
      dialect_4 = factor(dialect_4, levels = dialect_levels)
    )

  dialect_int <- as.integer(df_d$dialect_4)
  D           <- length(dialect_levels)

  # Integrity checks
  stopifnot(
    length(dialect_int) == dat_base$N,
    all(dialect_int >= 1L & dialect_int <= D),
    !anyNA(dialect_int)
  )

  # Report "other" assignments
  n_other <- sum(df_d$dialect_4 == "other")
  if (n_other > 0) {
    raw_other <- sort(unique(df_d$dialect_group[df_d$dialect_4 == "other"]))
    message(sprintf(
      "%d inscription(s) assigned to 'other' (raw dialect_group values: %s).",
      n_other, paste(raw_other, collapse = ", ")
    ))
  }

  list(
    D             = D,
    dialect       = dialect_int,
    dialect_levels = dialect_levels
  )
}

dial_aug <- build_dialect_augment(
  dat_base    = dat_stgp_ctx,
  t0          = t0_year,
  scale_years = t_scale_years
)

dialect_levels_cd <- dial_aug$dialect_levels

# Augment dat_stgp_ctx with dialect fields
dat_cd <- c(
  dat_stgp_ctx,
  list(
    D       = dial_aug$D,
    dialect = dial_aug$dialect
  )
)

# Final integrity checks on assembled data
stopifnot(
  dat_cd$N  == dat_stgp_ctx$N,
  dat_cd$S  == dat_stgp_ctx$S,
  dat_cd$T  == dat_stgp_ctx$T,
  dat_cd$K_u == dat_stgp_ctx$K_u,
  length(dat_cd$dialect) == dat_cd$N,
  dat_cd$D  == length(dialect_levels_cd)
)

message(sprintf(
  "Data assembled: N=%d, S=%d, T=%d, D=%d dialect groups, K_u=%d nodes.",
  dat_cd$N, dat_cd$S, dat_cd$T, dat_cd$D, dat_cd$K_u
))

# ==========================================================
#   COMPILE & SAMPLE
# ==========================================================

stan_file_cd <- file.path(
  here::here("stan"),
  paste0(fit_name_cd, ".stan")
)
if (!file.exists(path.expand(stan_file_cd))) stop("Stan file not found: ", stan_file_cd)
mod_cd <- cmdstanr::cmdstan_model(stan_file_cd)

fit_bb_tg_ctx_stgp_ctx_dial_stz <- mod_cd$sample(
  data             = dat_cd,
  output_dir       = csv_dir_cd,
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
if (has_var(fit_bb_tg_ctx_stgp_ctx_dial_stz, "log_lik")) {
  ll_mat_cd <- fit_bb_tg_ctx_stgp_ctx_dial_stz$draws("log_lik", format = "matrix")

  if (is.matrix(ll_mat_cd) && ncol(ll_mat_cd) > 20000) {
    set.seed(1); keep_cd <- sort(sample.int(ncol(ll_mat_cd), 20000))
    ll_mat_cd <- ll_mat_cd[, keep_cd, drop = FALSE]
  }

  if (is.matrix(ll_mat_cd) && ncol(ll_mat_cd) > 0) {
    loo_obj_cd <- loo::loo(ll_mat_cd, save_psis = TRUE, moment_match = FALSE)

    est_cd <- loo_obj_cd$estimates
    txt_cd <- paste0(
      "elpd_loo = ", sprintf("%.3f", est_cd["elpd_loo", "Estimate"]),
      " (SE = ", sprintf("%.3f", est_cd["elpd_loo", "SE"]), ")\n",
      "p_loo    = ", sprintf("%.3f", est_cd["p_loo",    "Estimate"]), "\n",
      "looic    = ", sprintf("%.3f", est_cd["looic",     "Estimate"]), "\n",
      "---------\n",
      "Pareto k diagnostics:\n",
      paste(capture.output(print(loo::pareto_k_table(loo_obj_cd))), collapse = "\n"), "\n"
    )
    writeLines(txt_cd, file.path(out_dir_cd, "loo.txt"))

    readr::write_csv(
      tibble::tibble(
        metric   = rownames(est_cd),
        estimate = as.numeric(est_cd[, "Estimate"]),
        se       = as.numeric(est_cd[, "SE"])
      ),
      file.path(out_dir_cd, "loo_estimates.csv")
    )

    pk_cd <- loo::pareto_k_table(loo_obj_cd)
    suppressWarnings(
      readr::write_csv(as.data.frame(pk_cd), file.path(out_dir_cd, "pareto_k.csv"))
    )

    saveRDS(loo_obj_cd, file.path(out_dir_cd, "loo.rds"))
  } else {
    readr::write_lines(
      "Found 'log_lik' but it contained 0 columns.",
      file.path(out_dir_cd, "loo.txt")
    )
  }
} else {
  readr::write_lines(
    "log_lik not found in fitted draws; LOO not computed.",
    file.path(out_dir_cd, "loo.txt")
  )
}

# ---- Save fit object ----
saveRDS(fit_bb_tg_ctx_stgp_ctx_dial_stz,
        file.path(out_dir_cd, paste0(fit_name_cd, ".rds")))

# ---- Index tables ----
tibble::tibble(type_id = seq_along(type_levels), type = type_levels) |>
  readr::write_csv(file.path(out_dir_cd, "type_index.csv"))

tibble::tibble(
  dialect_id = seq_along(dialect_levels_cd),
  dialect    = dialect_levels_cd
) |>
  readr::write_csv(file.path(out_dir_cd, "dialect_index.csv"))

sites_tbl |>
  dplyr::mutate(site_id = dplyr::row_number()) |>
  readr::write_csv(file.path(out_dir_cd, "site_index.csv"))

tibble::tibble(
  t0_year       = time_info$t0,
  t_scale_years = time_info$scale_years,
  K_u           = dat_cd$K_u,
  coord_tol     = time_info$tol
) |>
  readr::write_csv(file.path(out_dir_cd, "time_config.csv"))

# ---- Full parameter summary ----
# ---- Full parameter summary (exclude K_site; loads all other draws safely) ----
{
  md_vars_cd   <- fit_bb_tg_ctx_stgp_ctx_dial_stz$metadata()$stan_variables
  vars_to_summ <- md_vars_cd[!grepl("^K_site$", md_vars_cd)]
  summ_all_cd  <- fit_bb_tg_ctx_stgp_ctx_dial_stz$summary(variables = vars_to_summ)
  readr::write_csv(as.data.frame(summ_all_cd),
                   file.path(out_dir_cd, "summary_all.csv"))
  rm(summ_all_cd); gc()
}

# ---- Core scalar parameters (same set as Model 1) ----
core_vars_cd <- c("alpha", "beta_date", "gamma_ctx", "phi",
                  "sigma_gp", "rho_s", "rho_t")
summ_core_cd <- tryCatch(
  fit_bb_tg_ctx_stgp_ctx_dial_stz$summary(variables = core_vars_cd),
  error = function(e) tibble::tibble(note = conditionMessage(e))
)
readr::write_csv(as.data.frame(summ_core_cd),
                 file.path(out_dir_cd, "summary_core.csv"))

# ---- Dialect EIR deviations (sum-to-zero; delta_ctx[1..D]) ----
# delta_ctx[d] is the signed deviation of dialect d from gamma_ctx.
delta_draws_cd <- tryCatch(
  posterior::as_draws_df(
    fit_bb_tg_ctx_stgp_ctx_dial_stz$draws(variables = "delta_ctx")
  ),
  error = function(e) NULL
)

if (!is.null(delta_draws_cd)) {
  dc_cols <- grep("^delta_ctx\\[\\d+\\]$", names(delta_draws_cd), value = TRUE)
  idxs_dc <- as.integer(gsub(".*\\[(\\d+)\\]$", "\\1", dc_cols))
  dc_cols <- dc_cols[order(idxs_dc)]

  summ_delta_ctx_cd <- tibble::tibble(
    variable   = dc_cols,
    dialect_id = seq_along(dc_cols),
    dialect    = dialect_levels_cd[seq_along(dc_cols)],
    median     = sapply(dc_cols, \(nm) stats::median(delta_draws_cd[[nm]])),
    lo95       = sapply(dc_cols, \(nm) stats::quantile(delta_draws_cd[[nm]], 0.025)),
    hi95       = sapply(dc_cols, \(nm) stats::quantile(delta_draws_cd[[nm]], 0.975))
  )
  readr::write_csv(summ_delta_ctx_cd,
                   file.path(out_dir_cd, "summary_delta_ctx.csv"))
  rm(delta_draws_cd); gc()
} else {
  message("delta_ctx not found in draws; summary_delta_ctx.csv not written.")
}

# ---- Total dialect EIR shift (delta_ctx_total[1..D] = gamma_ctx + delta_ctx[d]) ----
# Primary comparison quantity against Model 2's delta_site[S].
# Sanity check: doric >> attic > other > ionic.
if (has_var(fit_bb_tg_ctx_stgp_ctx_dial_stz, "delta_ctx_total")) {
  summ_delta_total_cd <- fit_bb_tg_ctx_stgp_ctx_dial_stz$summary(
    variables = "delta_ctx_total"
  ) %>%
    dplyr::mutate(
      dialect_id = as.integer(
        gsub("^delta_ctx_total\\[(\\d+)\\]$", "\\1", variable)
      ),
      dialect = dialect_levels_cd[dialect_id]
    ) %>%
    dplyr::relocate(dialect_id, dialect, .after = variable)
  readr::write_csv(as.data.frame(summ_delta_total_cd),
                   file.path(out_dir_cd, "summary_delta_ctx_total.csv"))
} else {
  message("delta_ctx_total not found in draws; summary_delta_ctx_total.csv not written.")
}

# ---- Type fixed effects (sum-to-zero; beta_type[1..T]) ----
type_draws_cd <- tryCatch(
  posterior::as_draws_df(
    fit_bb_tg_ctx_stgp_ctx_dial_stz$draws(variables = "beta_type")
  ),
  error = function(e) NULL
)

if (!is.null(type_draws_cd)) {
  bt_cols_cd <- grep("^beta_type\\[\\d+\\]$", names(type_draws_cd), value = TRUE)
  idxs_cd    <- as.integer(gsub(".*\\[(\\d+)\\]$", "\\1", bt_cols_cd))
  bt_cols_cd <- bt_cols_cd[order(idxs_cd)]

  summ_type_cd <- tibble::tibble(
    variable = bt_cols_cd,
    type_id  = seq_along(bt_cols_cd),
    type     = type_levels[seq_along(bt_cols_cd)],
    median   = sapply(bt_cols_cd, \(nm) stats::median(type_draws_cd[[nm]])),
    lo95     = sapply(bt_cols_cd, \(nm) stats::quantile(type_draws_cd[[nm]], 0.025)),
    hi95     = sapply(bt_cols_cd, \(nm) stats::quantile(type_draws_cd[[nm]], 0.975))
  )
  readr::write_csv(summ_type_cd,
                   file.path(out_dir_cd, "summary_beta_type_fixed.csv"))
  rm(type_draws_cd); gc()
} else {
  message("beta_type not found in draws; summary_beta_type_fixed.csv not written.")
}

# ---- GP site-level effects (f_site[1..S]) ----
if (has_var(fit_bb_tg_ctx_stgp_ctx_dial_stz, "f_site")) {
  summ_fsite_cd <- fit_bb_tg_ctx_stgp_ctx_dial_stz$summary(variables = "f_site") %>%
    dplyr::mutate(
      site_id = as.integer(gsub("^f_site\\[(\\d+)\\]$", "\\1", variable))
    ) %>%
    dplyr::left_join(
      sites_tbl %>% dplyr::mutate(site_id = dplyr::row_number()),
      by = "site_id"
    ) %>%
    dplyr::relocate(site_id, latitude_site, longitude_site, .after = variable)
  readr::write_csv(as.data.frame(summ_fsite_cd),
                   file.path(out_dir_cd, "summary_f_site.csv"))
}

# ---- theta_bar quantities ----
for (qty in c("theta_bar", "theta_bar_out", "theta_bar_in")) {
  if (has_var(fit_bb_tg_ctx_stgp_ctx_dial_stz, qty)) {
    th_cd <- fit_bb_tg_ctx_stgp_ctx_dial_stz$summary(variables = qty) %>%
      dplyr::mutate(
        row = as.integer(gsub(paste0("^", qty, "\\[(\\d+)\\]$"), "\\1", variable))
      ) %>%
      dplyr::relocate(row, .before = 1)
    readr::write_csv(as.data.frame(th_cd),
                     file.path(out_dir_cd, paste0("summary_", qty, ".csv")))
  }
}

# ---- Diagnostics ----
diag_df_cd <- tryCatch(
  as.data.frame(fit_bb_tg_ctx_stgp_ctx_dial_stz$diagnostic_summary()),
  error = function(e) NULL
)
if (!is.null(diag_df_cd))
  readr::write_csv(diag_df_cd, file.path(out_dir_cd, "diagnostic_summary.csv"))

# ---- Provenance ----
run_meta_cd <- list(
  model_file    = stan_file_cd,
  fit_name      = fit_name_cd,
  results_dir   = normalizePath(out_dir_cd),
  N             = dat_cd$N,
  S             = dat_cd$S,
  T             = dat_cd$T,
  D             = dat_cd$D,
  K_u           = dat_cd$K_u,
  chains        = 4L,
  iter_warmup   = 2000L,
  iter_sampling = 2000L,
  adapt_delta   = 0.95,
  seed          = 2025L
)
readr::write_file(
  jsonlite::toJSON(run_meta_cd, auto_unbox = TRUE, pretty = TRUE),
  file.path(out_dir_cd, "run_metadata.json")
)

# ==========================================================
#   CONSOLE SUMMARY
# ==========================================================

message("\nKey scalar parameters:")
print(tibble::as_tibble(summ_core_cd))

message("\nDialect EIR deviations (sum-to-zero delta_ctx):")
if (exists("summ_delta_ctx_cd")) print(summ_delta_ctx_cd) else message("Not available.")

message("\nTotal dialect EIR shift (delta_ctx_total = gamma_ctx + delta_ctx):")
if (exists("summ_delta_total_cd")) print(tibble::as_tibble(summ_delta_total_cd)) else message("Not available.")

message("\nType fixed effects (sum-to-zero beta_type):")
if (exists("summ_type_cd")) print(summ_type_cd) else message("Not available.")

message("\nGP hyperparameters:")
print(tibble::as_tibble(
  fit_bb_tg_ctx_stgp_ctx_dial_stz$summary(variables = c("sigma_gp", "rho_s", "rho_t"))
))
