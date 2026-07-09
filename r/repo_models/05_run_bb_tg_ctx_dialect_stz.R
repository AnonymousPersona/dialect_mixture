# ==========================================================
#  05_run_bb_tg_ctx_dialect_stz.R
#
#  Dialect fixed-effects model: replaces the spatial GP with
#  four dialect-group intercepts (attic, doric, ionic, other)
#  and four dialect-group × EIR interactions.
#
#  Requires: 00_setup.R (provides dat_stgp_ctx, agg_data_dialect,
#            has_var, ensure_std_time, t0_year, t_scale_years, K_u)
# ==========================================================

source(here::here("r", "repo_models", "00_setup.R"))

# ==========================================================
#       bb_tg_ctx_dialect_stz  —  run + save
# ==========================================================

# ---------- paths ----------
fit_name_dialect <- "05_bb_tg_ctx_dialect_stz"
out_dir_dialect  <- file.path(results_root, fit_name_dialect)
dir.create(out_dir_dialect, recursive = TRUE, showWarnings = FALSE)

csv_dir_dialect <- file.path(out_dir_dialect, "csv")
dir.create(csv_dir_dialect, recursive = TRUE, showWarnings = FALSE)

# ==========================================================
#   DATA BUILDER — dialect fixed effects model
#   Requires dialect_group column; uses the same quadrature
#   and type encoding as build_data_stgp_ctx.
# ==========================================================

stopifnot(
  exists("has_var"),
  exists("ensure_std_time"),
  exists("t0_year"),
  exists("t_scale_years"),
  exists("K_u")
)

build_data_dialect <- function(
  df,
  K_u         = 11L,
  t0          = 0,
  scale_years = 100
) {
  need <- c("y1", "n1", "y2", "n2", "date_min", "date_max",
            "type", "dialect_group")
  miss <- setdiff(need, names(df))
  if (length(miss)) stop("Missing columns: ", paste(miss, collapse = ", "))

  # Time standardisation
  df_s <- ensure_std_time(df, t0 = t0, scale_years = scale_years)

  # Type levels: preserve factor ordering (funerary first)
  if (is.factor(df_s$type)) {
    type_levels <- levels(df_s$type)
  } else {
    df_s$type <- trimws(as.character(df_s$type))
    type_levels <- sort(unique(df_s$type))
    if ("funerary" %in% type_levels)
      type_levels <- c("funerary", setdiff(type_levels, "funerary"))
  }
  type_int <- as.integer(factor(as.character(df_s$type), levels = type_levels))

  # Dialect-group recoding: NA and anything outside {attic, doric, ionic}
  # is collapsed to "other".  Fixed alphabetical factor levels give stable
  # integer codes across runs: attic=1, doric=2, ionic=3, other=4.
  dialect_levels <- c("attic", "doric", "ionic", "other")
  df_s <- df_s %>%
    dplyr::mutate(
      dialect_4 = dplyr::case_when(
        is.na(dialect_group)              ~ "other",
        dialect_group == "attic"          ~ "attic",
        dialect_group == "doric"          ~ "doric",
        dialect_group == "ionic"          ~ "ionic",
        TRUE                              ~ "other"
      ),
      dialect_4 = factor(dialect_4, levels = dialect_levels)
    )

  dialect_int <- as.integer(df_s$dialect_4)

  # Quadrature nodes: uniform midpoints on (0, 1)
  u_grid <- (seq_len(K_u) - 0.5) / K_u

  data_list <- list(
    N         = nrow(df_s),
    y1        = as.integer(df_s$y1),
    n1        = as.integer(df_s$n1),
    y2        = as.integer(df_s$y2),
    n2        = as.integer(df_s$n2),
    K_u       = as.integer(K_u),
    u_grid    = as.numeric(u_grid),
    t_min_std = as.numeric(df_s$t_min_std),
    t_max_std = as.numeric(df_s$t_max_std),
    T         = length(type_levels),
    type      = type_int,
    D         = length(dialect_levels),
    dialect   = dialect_int
  )

  # Integrity checks
  stopifnot(all(data_list$type    >= 1 & data_list$type    <= data_list$T))
  stopifnot(all(data_list$dialect >= 1 & data_list$dialect <= data_list$D))
  stopifnot(
    lengths(data_list[c("y1","n1","y2","n2",
                         "t_min_std","t_max_std",
                         "type","dialect")]) == data_list$N
  )
  stopifnot(length(data_list$u_grid) == data_list$K_u)

  # Attach metadata as attributes (not sent to Stan)
  attr(data_list, ".levels") <- list(type    = type_levels,
                                     dialect = dialect_levels)
  attr(data_list, ".time")   <- list(t0 = t0, scale_years = scale_years)

  # Report "other" assignments so the user can inspect coverage
  n_other <- sum(df_s$dialect_4 == "other")
  if (n_other > 0) {
    raw_other <- sort(unique(df_s$dialect_group[df_s$dialect_4 == "other"]))
    message(sprintf(
      "%d inscription(s) assigned to 'other' dialect group (raw values: %s).",
      n_other, paste(raw_other, collapse = ", ")
    ))
  }

  data_list
}

# ==========================================================
#   BUILD STAN DATA
#   get_agg_data(require_dialect = TRUE) returns the same
#   inscription set as the STGP model but with dialect_group
#   and dialect columns included.
# ==========================================================

dat_dialect <- build_data_dialect(
  df          = get_agg_data(require_dialect = TRUE),
  K_u         = K_u,
  t0          = t0_year,
  scale_years = t_scale_years
)

# Recover metadata
type_levels_d    <- attr(dat_dialect, ".levels")$type
dialect_levels_d <- attr(dat_dialect, ".levels")$dialect
time_info_d      <- attr(dat_dialect, ".time")

# Sanity check: inscription count should match the STGP model
if (exists("dat_stgp_ctx") && dat_dialect$N != dat_stgp_ctx$N) {
  warning(sprintf(
    "N mismatch: dialect data has %d inscriptions, STGP data has %d.",
    dat_dialect$N, dat_stgp_ctx$N
  ))
}

message(sprintf(
  "Data assembled: N=%d inscriptions, T=%d types, D=%d dialect groups, K_u=%d nodes.",
  dat_dialect$N, dat_dialect$T, dat_dialect$D, dat_dialect$K_u
))

# ==========================================================
#   COMPILE & SAMPLE
# ==========================================================

stan_file_dialect <- file.path(
  here::here("stan"),
  paste0(fit_name_dialect, ".stan")
)
if (!file.exists(path.expand(stan_file_dialect)))
  stop("Stan file not found: ", stan_file_dialect)
mod_dialect <- cmdstanr::cmdstan_model(stan_file_dialect)

fit_bb_tg_ctx_dialect_stz <- mod_dialect$sample(
  data             = dat_dialect,
  output_dir       = csv_dir_dialect,
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
if (has_var(fit_bb_tg_ctx_dialect_stz, "log_lik")) {
  ll_mat_d <- fit_bb_tg_ctx_dialect_stz$draws("log_lik", format = "matrix")

  if (is.matrix(ll_mat_d) && ncol(ll_mat_d) > 20000) {
    set.seed(1); keep_d <- sort(sample.int(ncol(ll_mat_d), 20000))
    ll_mat_d <- ll_mat_d[, keep_d, drop = FALSE]
  }

  if (is.matrix(ll_mat_d) && ncol(ll_mat_d) > 0) {
    loo_obj_d <- loo::loo(ll_mat_d, save_psis = TRUE, moment_match = FALSE)

    est_d <- loo_obj_d$estimates
    txt_d <- paste0(
      "elpd_loo = ", sprintf("%.3f", est_d["elpd_loo", "Estimate"]),
      " (SE = ", sprintf("%.3f", est_d["elpd_loo", "SE"]), ")\n",
      "p_loo    = ", sprintf("%.3f", est_d["p_loo",    "Estimate"]), "\n",
      "looic    = ", sprintf("%.3f", est_d["looic",     "Estimate"]), "\n",
      "---------\n",
      "Pareto k diagnostics:\n",
      paste(capture.output(print(loo::pareto_k_table(loo_obj_d))), collapse = "\n"), "\n"
    )
    writeLines(txt_d, file.path(out_dir_dialect, "loo.txt"))

    readr::write_csv(
      tibble::tibble(
        metric   = rownames(est_d),
        estimate = as.numeric(est_d[, "Estimate"]),
        se       = as.numeric(est_d[, "SE"])
      ),
      file.path(out_dir_dialect, "loo_estimates.csv")
    )

    pk_d <- loo::pareto_k_table(loo_obj_d)
    suppressWarnings(
      readr::write_csv(as.data.frame(pk_d), file.path(out_dir_dialect, "pareto_k.csv"))
    )

    saveRDS(loo_obj_d, file.path(out_dir_dialect, "loo.rds"))
  } else {
    readr::write_lines(
      "Found 'log_lik' but it contained 0 columns.",
      file.path(out_dir_dialect, "loo.txt")
    )
  }
} else {
  readr::write_lines(
    "log_lik not found in fitted draws; LOO not computed.",
    file.path(out_dir_dialect, "loo.txt")
  )
}

# ---- Save fit object ----
saveRDS(fit_bb_tg_ctx_dialect_stz,
        file.path(out_dir_dialect, paste0(fit_name_dialect, ".rds")))

# ---- Index tables ----
tibble::tibble(type_id = seq_along(type_levels_d), type = type_levels_d) |>
  readr::write_csv(file.path(out_dir_dialect, "type_index.csv"))

tibble::tibble(dialect_id = seq_along(dialect_levels_d),
               dialect    = dialect_levels_d) |>
  readr::write_csv(file.path(out_dir_dialect, "dialect_index.csv"))

tibble::tibble(
  t0_year       = time_info_d$t0,
  t_scale_years = time_info_d$scale_years,
  K_u           = dat_dialect$K_u
) |>
  readr::write_csv(file.path(out_dir_dialect, "time_config.csv"))

# ---- Full parameter summary ----
summ_all_d <- fit_bb_tg_ctx_dialect_stz$summary()
readr::write_csv(as.data.frame(summ_all_d),
                 file.path(out_dir_dialect, "summary_all.csv"))

# ---- Core scalar parameters ----
core_vars_d <- c("alpha", "beta_date", "gamma_ctx", "phi")
summ_core_d <- tryCatch(
  fit_bb_tg_ctx_dialect_stz$summary(variables = core_vars_d),
  error = function(e) tibble::tibble(note = conditionMessage(e))
)
readr::write_csv(as.data.frame(summ_core_d),
                 file.path(out_dir_dialect, "summary_core.csv"))

# ---- Type fixed effects (sum-to-zero; beta_type[1..T]) ----
type_draws_d <- tryCatch(
  posterior::as_draws_df(
    fit_bb_tg_ctx_dialect_stz$draws(variables = "beta_type")
  ),
  error = function(e) NULL
)

if (!is.null(type_draws_d)) {
  bt_cols_d <- grep("^beta_type\\[\\d+\\]$", names(type_draws_d), value = TRUE)
  idxs_d    <- as.integer(gsub(".*\\[(\\d+)\\]$", "\\1", bt_cols_d))
  bt_cols_d <- bt_cols_d[order(idxs_d)]

  summ_type_d <- tibble::tibble(
    variable = bt_cols_d,
    type_id  = seq_along(bt_cols_d),
    type     = type_levels_d[seq_along(bt_cols_d)],
    median   = sapply(bt_cols_d, \(nm) stats::median(type_draws_d[[nm]])),
    lo95     = sapply(bt_cols_d, \(nm) stats::quantile(type_draws_d[[nm]], 0.025)),
    hi95     = sapply(bt_cols_d, \(nm) stats::quantile(type_draws_d[[nm]], 0.975))
  )
  readr::write_csv(summ_type_d,
                   file.path(out_dir_dialect, "summary_beta_type_fixed.csv"))
} else {
  message("beta_type not found in draws; summary_beta_type_fixed.csv not written.")
}

# ---- Dialect fixed effects (sum-to-zero; beta_dialect[1..D]) ----
dialect_draws_d <- tryCatch(
  posterior::as_draws_df(
    fit_bb_tg_ctx_dialect_stz$draws(variables = "beta_dialect")
  ),
  error = function(e) NULL
)

if (!is.null(dialect_draws_d)) {
  bd_cols   <- grep("^beta_dialect\\[\\d+\\]$", names(dialect_draws_d), value = TRUE)
  idxs_bd   <- as.integer(gsub(".*\\[(\\d+)\\]$", "\\1", bd_cols))
  bd_cols   <- bd_cols[order(idxs_bd)]

  summ_dialect_d <- tibble::tibble(
    variable   = bd_cols,
    dialect_id = seq_along(bd_cols),
    dialect    = dialect_levels_d[seq_along(bd_cols)],
    median     = sapply(bd_cols, \(nm) stats::median(dialect_draws_d[[nm]])),
    lo95       = sapply(bd_cols, \(nm) stats::quantile(dialect_draws_d[[nm]], 0.025)),
    hi95       = sapply(bd_cols, \(nm) stats::quantile(dialect_draws_d[[nm]], 0.975))
  )
  readr::write_csv(summ_dialect_d,
                   file.path(out_dir_dialect, "summary_beta_dialect_fixed.csv"))
} else {
  message("beta_dialect not found in draws; summary_beta_dialect_fixed.csv not written.")
}

# ---- theta_bar quantities ----
for (qty in c("theta_bar", "theta_bar_out", "theta_bar_in")) {
  if (has_var(fit_bb_tg_ctx_dialect_stz, qty)) {
    th_d <- fit_bb_tg_ctx_dialect_stz$summary(variables = qty) %>%
      dplyr::mutate(
        row = as.integer(gsub(paste0("^", qty, "\\[(\\d+)\\]$"), "\\1", variable))
      ) %>%
      dplyr::relocate(row, .before = 1)
    readr::write_csv(as.data.frame(th_d),
                     file.path(out_dir_dialect, paste0("summary_", qty, ".csv")))
  }
}

# ---- Diagnostics ----
diag_df_d <- tryCatch(
  as.data.frame(fit_bb_tg_ctx_dialect_stz$diagnostic_summary()),
  error = function(e) NULL
)
if (!is.null(diag_df_d))
  readr::write_csv(diag_df_d, file.path(out_dir_dialect, "diagnostic_summary.csv"))

# ---- Provenance ----
run_meta_d <- list(
  model_file    = stan_file_dialect,
  fit_name      = fit_name_dialect,
  results_dir   = normalizePath(out_dir_dialect),
  N             = dat_dialect$N,
  T             = dat_dialect$T,
  D             = dat_dialect$D,
  K_u           = dat_dialect$K_u,
  chains        = 4L,
  iter_warmup   = 2000L,
  iter_sampling = 2000L,
  adapt_delta   = 0.90,
  seed          = 2025L
)
readr::write_file(
  jsonlite::toJSON(run_meta_d, auto_unbox = TRUE, pretty = TRUE),
  file.path(out_dir_dialect, "run_metadata.json")
)

# ==========================================================
#   CONSOLE SUMMARY
# ==========================================================

message("\nCore scalar parameters:")
print(tibble::as_tibble(summ_core_d))

message("\nType fixed effects (sum-to-zero beta_type):")
if (exists("summ_type_d")) print(summ_type_d) else message("Not available.")

message("\nDialect fixed effects (sum-to-zero beta_dialect):")
if (exists("summ_dialect_d")) print(summ_dialect_d) else message("Not available.")
