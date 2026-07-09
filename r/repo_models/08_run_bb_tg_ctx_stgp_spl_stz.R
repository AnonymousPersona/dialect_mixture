# ==========================================================
#  08_run_bb_tg_ctx_stgp_spl_stz.R
#
#  Nonlinear date model: replaces beta_date * t with a cubic
#  B-spline temporal trend, keeping the STGP.  Tests whether
#  the koine spread is nonlinear.
#
#  Requires: 00_setup.R (provides dat_stgp_ctx, sites_tbl,
#            type_levels, time_info, has_var)
# ==========================================================

source(here::here("r", "repo_models", "00_setup.R"))

# ==========================================================
#       bb_tg_ctx_stgp_spl_stz  —  run + save
#
#  Nonlinear date model: replaces beta_date * t with a
#  cubic B-spline temporal trend, keeping the STGP.
#  B-spline basis is evaluated at every quadrature point
#  in R and passed to Stan as B_quad[N, K_u, Q].
#  u_grid is NOT passed to Stan (basis already evaluated).
# ==========================================================

stopifnot(
  exists("dat_stgp_ctx"),
  exists("has_var"),
  exists("type_levels"),
  exists("sites_tbl"),
  exists("time_info")
)

# ---------- paths ----------
fit_name_spl     <- "08_bb_tg_ctx_stgp_spl_stz"
out_dir_spl      <- file.path(results_root, fit_name_spl)
dir.create(out_dir_spl, recursive = TRUE, showWarnings = FALSE)

csv_dir_spl <- file.path(out_dir_spl, "csv")
dir.create(csv_dir_spl, recursive = TRUE, showWarnings = FALSE)

# ==========================================================
#   B-SPLINE BASIS CONSTRUCTION
#
#   For each inscription i and quadrature node k, evaluates
#   the cubic B-spline basis at t_ik = t_min[i] + u[k]*dt[i].
#   Returns an array[N, K_u, Q] suitable for CmdStanR where
#   arr[i, k, q] = q-th basis function at quadrature point k
#   for inscription i.
#
#   Column-centring (centre = TRUE):
#     For each basis function q, subtract the mean value
#     across all N * K_u evaluation points.  This ensures
#     the spline contributes zero net shift to the predictor
#     mean, keeping alpha as the grand-mean intercept —
#     the same identification used for beta_type.
#
#   n_knots_int = 3 (default) gives Q = 7 basis functions.
#   n_knots_int = 4 gives Q = 8.  Interior knots are placed
#   at equally-spaced quantiles of the inscription midpoints.
#   Boundary knots span the full range of all t_ik values so
#   that no quadrature point falls outside the support.
# ==========================================================

build_bspline_basis <- function(t_min_std,
                                t_max_std,
                                K_u,
                                n_knots_int = 3L,
                                centre      = TRUE) {
  stopifnot(
    length(t_min_std) == length(t_max_std),
    all(t_min_std <= t_max_std),
    K_u         >= 1L,
    n_knots_int >= 1L
  )

  N_loc  <- length(t_min_std)
  dt_loc <- t_max_std - t_min_std

  # Quadrature nodes on (0, 1) — same formula as build_data_stgp_ctx.
  # Kept local: NOT passed to Stan.
  u_grid_loc <- (seq_len(K_u) - 0.5) / K_u

  # Interior knot positions from quantiles of inscription midpoints
  t_mid_ref  <- (t_min_std + t_max_std) / 2
  knot_probs <- seq(0, 1, length.out = n_knots_int + 2L)[seq(2L, n_knots_int + 1L)]
  knots_int  <- as.numeric(quantile(t_mid_ref, probs = knot_probs))

  # Boundary knots: full range of ALL t_ik across every (i, k) pair,
  # ensuring no quadrature point falls outside the spline support.
  t_all <- as.numeric(
    outer(t_min_std, u_grid_loc * dt_loc, "+")  # N × K_u matrix
  )
  boundary <- c(min(t_all), max(t_all))

  # Q: n_knots_int interior knots + degree 3 + 1 intercept = n_knots_int + 4
  Q_loc <- n_knots_int + 4L

  # Verify dimension against a trial evaluation
  bs_trial <- splines::bs(t_mid_ref,
                          knots          = knots_int,
                          degree         = 3L,
                          intercept      = TRUE,
                          Boundary.knots = boundary)
  stopifnot(ncol(bs_trial) == Q_loc)

  # Build B_quad[N, K_u, Q] -------------------------------------------
  # arr[i, k, q] = q-th basis function evaluated at t_ik.
  # Loop over k: evaluate basis at the N-vector t_ik, store as
  # the [, k, ] slice of the array.
  B_quad_arr <- array(0, dim = c(N_loc, K_u, Q_loc))

  for (k in seq_len(K_u)) {
    t_ik  <- t_min_std + u_grid_loc[k] * dt_loc   # length N
    B_mat <- splines::bs(t_ik,
                         knots          = knots_int,
                         degree         = 3L,
                         intercept      = TRUE,
                         Boundary.knots = boundary)
    # B_mat is N × Q_loc; store in the k-th quadrature slice
    B_quad_arr[, k, ] <- B_mat
  }

  # Column-centre: subtract the grand mean of each basis function
  # across all N * K_u evaluation points.
  if (centre) {
    for (q in seq_len(Q_loc)) {
      B_quad_arr[, , q] <- B_quad_arr[, , q] - mean(B_quad_arr[, , q])
    }
  }

  list(
    B_quad      = B_quad_arr,        # array[N, K_u, Q]
    Q           = Q_loc,             # number of basis functions
    knots_int   = knots_int,         # interior knot positions
    boundary    = boundary,          # boundary knot positions
    n_knots_int = n_knots_int,
    u_grid      = u_grid_loc,        # stored for metadata only
    col_means   = if (centre) {
                    vapply(seq_len(Q_loc), \(q) mean(B_quad_arr[, , q] +
                           mean(B_quad_arr[, , q])), numeric(1L))
                  } else rep(0, Q_loc)
  )
}

# ==========================================================
#   BUILD B-SPLINE BASIS
# ==========================================================

spl_basis <- build_bspline_basis(
  t_min_std   = dat_stgp_ctx$t_min_std,
  t_max_std   = dat_stgp_ctx$t_max_std,
  K_u         = dat_stgp_ctx$K_u,
  n_knots_int = 3L,      # Q = 7; change to 4L for Q = 8
  centre      = TRUE
)

message(sprintf(
  "B-spline basis: Q=%d basis functions, %d interior knots, boundary [%.3f, %.3f].",
  spl_basis$Q, spl_basis$n_knots_int,
  spl_basis$boundary[1], spl_basis$boundary[2]
))

# ==========================================================
#   BUILD STAN DATA
#   Start from dat_stgp_ctx, remove u_grid (not declared in
#   the spline model's data block), add Q and B_quad.
# ==========================================================

dat_spl <- c(
  dat_stgp_ctx[setdiff(names(dat_stgp_ctx), "u_grid")],
  list(
    Q      = spl_basis$Q,
    B_quad = spl_basis$B_quad    # array[N, K_u, Q]
  )
)

# Integrity checks
stopifnot(
  dat_spl$N  == dat_stgp_ctx$N,
  dat_spl$S  == dat_stgp_ctx$S,
  dat_spl$T  == dat_stgp_ctx$T,
  dat_spl$K_u == dat_stgp_ctx$K_u,
  is.array(dat_spl$B_quad),
  identical(dim(dat_spl$B_quad), c(dat_spl$N, dat_spl$K_u, dat_spl$Q))
)

message(sprintf(
  "Stan data assembled: N=%d, S=%d, T=%d, K_u=%d, Q=%d. B_quad dim: [%s].",
  dat_spl$N, dat_spl$S, dat_spl$T, dat_spl$K_u, dat_spl$Q,
  paste(dim(dat_spl$B_quad), collapse = ", ")
))

# ==========================================================
#   COMPILE & SAMPLE
# ==========================================================

stan_file_spl <- file.path(
  here::here("stan"),
  paste0(fit_name_spl, ".stan")
)
if (!file.exists(path.expand(stan_file_spl))) stop("Stan file not found: ", stan_file_spl)
mod_spl <- cmdstanr::cmdstan_model(stan_file_spl)

fit_bb_tg_ctx_stgp_spl_stz <- mod_spl$sample(
  data             = dat_spl,
  output_dir       = csv_dir_spl,
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
if (has_var(fit_bb_tg_ctx_stgp_spl_stz, "log_lik")) {
  ll_mat_spl <- fit_bb_tg_ctx_stgp_spl_stz$draws("log_lik", format = "matrix")

  if (is.matrix(ll_mat_spl) && ncol(ll_mat_spl) > 20000) {
    set.seed(1); keep_spl <- sort(sample.int(ncol(ll_mat_spl), 20000))
    ll_mat_spl <- ll_mat_spl[, keep_spl, drop = FALSE]
  }

  if (is.matrix(ll_mat_spl) && ncol(ll_mat_spl) > 0) {
    loo_obj_spl <- loo::loo(ll_mat_spl, save_psis = TRUE, moment_match = FALSE)

    est_spl <- loo_obj_spl$estimates
    txt_spl <- paste0(
      "elpd_loo = ", sprintf("%.3f", est_spl["elpd_loo", "Estimate"]),
      " (SE = ", sprintf("%.3f", est_spl["elpd_loo", "SE"]), ")\n",
      "p_loo    = ", sprintf("%.3f", est_spl["p_loo",    "Estimate"]), "\n",
      "looic    = ", sprintf("%.3f", est_spl["looic",     "Estimate"]), "\n",
      "---------\n",
      "Pareto k diagnostics:\n",
      paste(capture.output(print(loo::pareto_k_table(loo_obj_spl))), collapse = "\n"), "\n"
    )
    writeLines(txt_spl, file.path(out_dir_spl, "loo.txt"))

    readr::write_csv(
      tibble::tibble(
        metric   = rownames(est_spl),
        estimate = as.numeric(est_spl[, "Estimate"]),
        se       = as.numeric(est_spl[, "SE"])
      ),
      file.path(out_dir_spl, "loo_estimates.csv")
    )

    pk_spl <- loo::pareto_k_table(loo_obj_spl)
    suppressWarnings(
      readr::write_csv(as.data.frame(pk_spl), file.path(out_dir_spl, "pareto_k.csv"))
    )

    saveRDS(loo_obj_spl, file.path(out_dir_spl, "loo.rds"))
  } else {
    readr::write_lines(
      "Found 'log_lik' but it contained 0 columns.",
      file.path(out_dir_spl, "loo.txt")
    )
  }
} else {
  readr::write_lines(
    "log_lik not found in fitted draws; LOO not computed.",
    file.path(out_dir_spl, "loo.txt")
  )
}

# ---- Save fit object ----
saveRDS(fit_bb_tg_ctx_stgp_spl_stz,
        file.path(out_dir_spl, paste0(fit_name_spl, ".rds")))

# ---- Index tables ----
tibble::tibble(type_id = seq_along(type_levels), type = type_levels) |>
  readr::write_csv(file.path(out_dir_spl, "type_index.csv"))

sites_tbl |>
  dplyr::mutate(site_id = dplyr::row_number()) |>
  readr::write_csv(file.path(out_dir_spl, "site_index.csv"))

tibble::tibble(
  t0_year       = time_info$t0,
  t_scale_years = time_info$scale_years,
  K_u           = dat_spl$K_u,
  coord_tol     = time_info$tol
) |>
  readr::write_csv(file.path(out_dir_spl, "time_config.csv"))

# ---- Spline metadata (for posterior curve reconstruction) ----
tibble::tibble(
  q             = seq_len(spl_basis$Q),
  knot_position = c(spl_basis$knots_int,
                    rep(NA_real_, spl_basis$Q - spl_basis$n_knots_int))
) |>
  readr::write_csv(file.path(out_dir_spl, "spline_index.csv"))

tibble::tibble(
  n_knots_int   = spl_basis$n_knots_int,
  Q             = spl_basis$Q,
  degree        = 3L,
  boundary_lo   = spl_basis$boundary[1],
  boundary_hi   = spl_basis$boundary[2],
  centred       = TRUE
) |>
  readr::write_csv(file.path(out_dir_spl, "spline_config.csv"))

saveRDS(spl_basis, file.path(out_dir_spl, "spline_basis.rds"))

# ---- Full parameter summary (exclude K_site if present) ----
{
  md_vars_spl  <- fit_bb_tg_ctx_stgp_spl_stz$metadata()$stan_variables
  vars_to_summ <- md_vars_spl[!grepl("^K_site$", md_vars_spl)]
  summ_all_spl <- fit_bb_tg_ctx_stgp_spl_stz$summary(variables = vars_to_summ)
  readr::write_csv(as.data.frame(summ_all_spl),
                   file.path(out_dir_spl, "summary_all.csv"))
}

# ---- Core scalar parameters (no beta_date; GP hyperparameters retained) ----
core_vars_spl <- c("alpha", "gamma_ctx", "phi", "sigma_gp", "rho_s", "rho_t")
summ_core_spl <- tryCatch(
  fit_bb_tg_ctx_stgp_spl_stz$summary(variables = core_vars_spl),
  error = function(e) tibble::tibble(note = conditionMessage(e))
)
readr::write_csv(as.data.frame(summ_core_spl),
                 file.path(out_dir_spl, "summary_core.csv"))

# ---- B-spline temporal coefficients (beta_spl[1..Q]) ----
spl_draws <- tryCatch(
  posterior::as_draws_df(
    fit_bb_tg_ctx_stgp_spl_stz$draws(variables = "beta_spl")
  ),
  error = function(e) NULL
)

if (!is.null(spl_draws)) {
  bs_cols <- grep("^beta_spl\\[\\d+\\]$", names(spl_draws), value = TRUE)
  idxs_bs <- as.integer(gsub(".*\\[(\\d+)\\]$", "\\1", bs_cols))
  bs_cols <- bs_cols[order(idxs_bs)]

  summ_spl <- tibble::tibble(
    variable = bs_cols,
    q        = seq_along(bs_cols),
    median   = sapply(bs_cols, \(nm) stats::median(spl_draws[[nm]])),
    lo95     = sapply(bs_cols, \(nm) stats::quantile(spl_draws[[nm]], 0.025)),
    hi95     = sapply(bs_cols, \(nm) stats::quantile(spl_draws[[nm]], 0.975))
  )
  readr::write_csv(summ_spl, file.path(out_dir_spl, "summary_beta_spl.csv"))
} else {
  message("beta_spl not found in draws; summary_beta_spl.csv not written.")
}

# ---- Type fixed effects (sum-to-zero; beta_type[1..T]) ----
type_draws_spl <- tryCatch(
  posterior::as_draws_df(
    fit_bb_tg_ctx_stgp_spl_stz$draws(variables = "beta_type")
  ),
  error = function(e) NULL
)

if (!is.null(type_draws_spl)) {
  bt_cols_spl <- grep("^beta_type\\[\\d+\\]$", names(type_draws_spl), value = TRUE)
  idxs_spl    <- as.integer(gsub(".*\\[(\\d+)\\]$", "\\1", bt_cols_spl))
  bt_cols_spl <- bt_cols_spl[order(idxs_spl)]

  summ_type_spl <- tibble::tibble(
    variable = bt_cols_spl,
    type_id  = seq_along(bt_cols_spl),
    type     = type_levels[seq_along(bt_cols_spl)],
    median   = sapply(bt_cols_spl, \(nm) stats::median(type_draws_spl[[nm]])),
    lo95     = sapply(bt_cols_spl, \(nm) stats::quantile(type_draws_spl[[nm]], 0.025)),
    hi95     = sapply(bt_cols_spl, \(nm) stats::quantile(type_draws_spl[[nm]], 0.975))
  )
  readr::write_csv(summ_type_spl,
                   file.path(out_dir_spl, "summary_beta_type_fixed.csv"))
} else {
  message("beta_type not found in draws; summary_beta_type_fixed.csv not written.")
}

# ---- GP site-level effects (f_site[1..S]) ----
if (has_var(fit_bb_tg_ctx_stgp_spl_stz, "f_site")) {
  summ_fsite_spl <- fit_bb_tg_ctx_stgp_spl_stz$summary(variables = "f_site") %>%
    dplyr::mutate(
      site_id = as.integer(gsub("^f_site\\[(\\d+)\\]$", "\\1", variable))
    ) %>%
    dplyr::left_join(
      sites_tbl %>% dplyr::mutate(site_id = dplyr::row_number()),
      by = "site_id"
    ) %>%
    dplyr::relocate(site_id, latitude_site, longitude_site, .after = variable)
  readr::write_csv(as.data.frame(summ_fsite_spl),
                   file.path(out_dir_spl, "summary_f_site.csv"))
}

# ---- theta_bar quantities ----
for (qty in c("theta_bar", "theta_bar_out", "theta_bar_in")) {
  if (has_var(fit_bb_tg_ctx_stgp_spl_stz, qty)) {
    th_spl <- fit_bb_tg_ctx_stgp_spl_stz$summary(variables = qty) %>%
      dplyr::mutate(
        row = as.integer(gsub(paste0("^", qty, "\\[(\\d+)\\]$"), "\\1", variable))
      ) %>%
      dplyr::relocate(row, .before = 1)
    readr::write_csv(as.data.frame(th_spl),
                     file.path(out_dir_spl, paste0("summary_", qty, ".csv")))
  }
}

# ---- Diagnostics ----
diag_df_spl <- tryCatch(
  as.data.frame(fit_bb_tg_ctx_stgp_spl_stz$diagnostic_summary()),
  error = function(e) NULL
)
if (!is.null(diag_df_spl))
  readr::write_csv(diag_df_spl, file.path(out_dir_spl, "diagnostic_summary.csv"))

# ---- Provenance ----
run_meta_spl <- list(
  model_file    = stan_file_spl,
  fit_name      = fit_name_spl,
  results_dir   = normalizePath(out_dir_spl),
  N             = dat_spl$N,
  S             = dat_spl$S,
  T             = dat_spl$T,
  K_u           = dat_spl$K_u,
  Q             = dat_spl$Q,
  n_knots_int   = spl_basis$n_knots_int,
  chains        = 4L,
  iter_warmup   = 2000L,
  iter_sampling = 2000L,
  adapt_delta   = 0.95,
  seed          = 2025L
)
readr::write_file(
  jsonlite::toJSON(run_meta_spl, auto_unbox = TRUE, pretty = TRUE),
  file.path(out_dir_spl, "run_metadata.json")
)

# ==========================================================
#   CONSOLE SUMMARY
# ==========================================================

message("\nCore scalar parameters (no beta_date):")
print(tibble::as_tibble(summ_core_spl))

message("\nB-spline coefficients (beta_spl[1..Q]):")
if (exists("summ_spl")) print(summ_spl) else message("Not available.")

message("\nType fixed effects (sum-to-zero beta_type):")
if (exists("summ_type_spl")) print(summ_type_spl) else message("Not available.")

message("\nGP hyperparameters:")
print(tibble::as_tibble(
  fit_bb_tg_ctx_stgp_spl_stz$summary(variables = c("sigma_gp", "rho_s", "rho_t"))
))
