# ==========================================================
#  03_run_bb_tgm_ctx_stgp_ctx_attica_egypt.R
#
#  Attica/Egypt model: extends Model 01 with explicit soft
#  region-level EIR shifts for Attica and Egypt encoded via
#  unconstrained normal priors N(1.5, 1).
#
#  Requires: 00_setup.R (provides dat_stgp_ctx, agg_data,
#            sites_tbl, type_levels, time_info, has_var)
# ==========================================================

source(here::here("r", "repo_models", "00_setup.R"))

stopifnot(
  exists("dat_stgp_ctx"),
  exists("build_data_stgp_ctx"),
  exists("type_levels"),
  exists("sites_tbl"),
  exists("time_info"),
  exists("has_var"),
  exists("agg_data")           # populated by the aggregate-data chunk
)

# ----------------------------------------------------------
#   PATHS
# ----------------------------------------------------------

fit_name_ae <- "03_bb_tgm_ctx_stgp_ctx_attica_egypt"
out_dir_ae  <- file.path(results_root, fit_name_ae)
dir.create(out_dir_ae, recursive = TRUE, showWarnings = FALSE)

csv_dir_ae <- file.path(out_dir_ae, "csv")
dir.create(csv_dir_ae, recursive = TRUE, showWarnings = FALSE)

# ----------------------------------------------------------
#   BUILD REGION INDICATOR VECTORS
#
#   build_data_stgp_ctx() builds sites_tbl from coordinates
#   alone, so it carries no region column.  Region labels are
#   recovered here from agg_data (already in the session)
#   by replicating the same coordinate-rounding site-index
#   logic used inside the builder, then joining onto sites_tbl.
#
#   The rounding tolerance must match the tol argument that
#   was passed to build_data_stgp_ctx(); it is stored in
#   time_info$tol.
#
#   Each unique (lat_k, lon_k) pair maps to exactly one site.
#   Where multiple inscriptions share a site, the modal region
#   label is used (with a warning if the site is not
#   internally consistent, e.g. due to coordinate rounding
#   pulling two physically distinct locations together).
# ----------------------------------------------------------

stopifnot(
  "agg_data must contain columns 'latitude', 'longitude', 'region'" =
    all(c("latitude", "longitude", "region") %in% names(agg_data))
)

{
  tol <- time_info$tol   # e.g. 1e-5; same value used in build_data_stgp_ctx()

  round_to <- function(x, tol) round(x / tol) * tol

  # Build a (lat_k, lon_k) -> region lookup from the raw inscriptions.
  # If a site has inscriptions from more than one region (which should not
  # happen with well-formed data, but could arise from coordinate rounding),
  # take the modal value and warn.
  site_region_lut <- agg_data |>
    dplyr::filter(is.finite(latitude), is.finite(longitude)) |>
    dplyr::mutate(
      lat_k  = round_to(latitude,  tol),
      lon_k  = round_to(longitude, tol),
      region = trimws(region)
    ) |>
    dplyr::group_by(lat_k, lon_k) |>
    dplyr::summarise(
      region_vals = list(region),
      region      = {
        tbl <- sort(table(region), decreasing = TRUE)
        if (length(tbl) > 1L)
          warning(sprintf(
            "Site at (lat_k=%.6f, lon_k=%.6f) has %d region labels: %s. ",
            lat_k[1], lon_k[1], length(tbl),
            paste(names(tbl), collapse = ", ")
          ), "Using modal value '", names(tbl)[1], "'.",
          call. = FALSE)
        names(tbl)[1]
      },
      .groups = "drop"
    ) |>
    dplyr::select(lat_k, lon_k, region)

  # Round sites_tbl coordinates to the same tolerance to form the join key.
  # sites_tbl rows are already in site_id order (row number = site_id).
  sites_tbl_region <- sites_tbl |>
    dplyr::mutate(
      lat_k = round_to(latitude_site,  tol),
      lon_k = round_to(longitude_site, tol)
    ) |>
    dplyr::left_join(site_region_lut, by = c("lat_k", "lon_k")) |>
    dplyr::select(-lat_k, -lon_k)

  # Every site must have resolved to a region; NA means the rounding key
  # did not match anything in agg_data, which indicates a data inconsistency.
  unmatched <- which(is.na(sites_tbl_region$region))
  if (length(unmatched) > 0L)
    stop(
      length(unmatched), " site(s) in sites_tbl could not be matched to a ",
      "region in agg_data (site indices: ",
      paste(unmatched, collapse = ", "), "). ",
      "Check that agg_data and dat_stgp_ctx were built from the same data ",
      "with the same coordinate-rounding tolerance (time_info$tol = ", tol, ")."
    )

  region_lower <- tolower(sites_tbl_region$region)
}

attica_site <- as.integer(region_lower == "attica")
egypt_site  <- as.integer(region_lower == "egypt")

# Sanity checks before passing to Stan
if (!any(attica_site == 1L))
  warning(
    "No sites are coded as Attica (attica_site is all zeros). ",
    "Unique region values found: ",
    paste(sort(unique(region_lower)), collapse = ", "), ".",
    call. = FALSE
  )

if (!any(egypt_site == 1L))
  warning(
    "No sites are coded as Egypt (egypt_site is all zeros). ",
    "Unique region values found: ",
    paste(sort(unique(region_lower)), collapse = ", "), ".",
    call. = FALSE
  )

overlap <- which(attica_site == 1L & egypt_site == 1L)
if (length(overlap) > 0L)
  stop(
    "Sites ", paste(overlap, collapse = ", "),
    " appear in both attica_site and egypt_site. ",
    "Each site must belong to at most one region."
  )

message(sprintf(
  "Region indicators: %d Attica site(s), %d Egypt site(s), %d other site(s).",
  sum(attica_site), sum(egypt_site),
  sum(attica_site == 0L & egypt_site == 0L)
))

# ----------------------------------------------------------
#   ASSEMBLE DATA LIST
#
#   Extend dat_stgp_ctx with the two new indicators.
#   A fresh list is created so dat_stgp_ctx is unchanged
#   and remains available for other models.
# ----------------------------------------------------------

dat_ae <- c(
  dat_stgp_ctx,
  list(
    attica_site = attica_site,
    egypt_site  = egypt_site
  )
)

# ----------------------------------------------------------
#   COMPILE & SAMPLE
# ----------------------------------------------------------

stan_file_ae <- file.path(
  here::here("stan"),
  paste0(fit_name_ae, ".stan")
)
if (!file.exists(path.expand(stan_file_ae)))
  stop("Stan file not found: ", stan_file_ae)

mod_ae <- cmdstanr::cmdstan_model(stan_file_ae)

fit_bb_tgm_ctx_stgp_ctx_attica_egypt <- mod_ae$sample(
  data             = dat_ae,
  output_dir       = csv_dir_ae,
  seed             = 2025,
  chains           = 4, parallel_chains = 4,
  iter_warmup      = 2000, iter_sampling  = 2000,
  adapt_delta      = 0.95, max_treedepth  = 12,
  refresh          = 50
)

# ----------------------------------------------------------
#   LOO
# ----------------------------------------------------------

if (has_var(fit_bb_tgm_ctx_stgp_ctx_attica_egypt, "log_lik")) {

  ll_mat_ae <- fit_bb_tgm_ctx_stgp_ctx_attica_egypt$draws(
    "log_lik", format = "matrix"
  )

  # Subsample columns to ≤ 20 000 if necessary (PSIS is O(N))
  if (is.matrix(ll_mat_ae) && ncol(ll_mat_ae) > 20000L) {
    set.seed(1L)
    keep_ae   <- sort(sample.int(ncol(ll_mat_ae), 20000L))
    ll_mat_ae <- ll_mat_ae[, keep_ae, drop = FALSE]
  }

  if (is.matrix(ll_mat_ae) && ncol(ll_mat_ae) > 0L) {

    loo_ae <- loo::loo(ll_mat_ae, save_psis = TRUE, moment_match = FALSE)

    est_ae <- loo_ae$estimates
    txt_ae <- paste0(
      "elpd_loo = ", sprintf("%.3f", est_ae["elpd_loo", "Estimate"]),
      " (SE = ",     sprintf("%.3f", est_ae["elpd_loo", "SE"]), ")\n",
      "p_loo    = ", sprintf("%.3f", est_ae["p_loo",    "Estimate"]), "\n",
      "looic    = ", sprintf("%.3f", est_ae["looic",     "Estimate"]), "\n",
      "---------\n",
      "Pareto k diagnostics:\n",
      paste(capture.output(print(loo::pareto_k_table(loo_ae))), collapse = "\n"),
      "\n"
    )
    writeLines(txt_ae, file.path(out_dir_ae, "loo.txt"))

    readr::write_csv(
      tibble::tibble(
        metric   = rownames(est_ae),
        estimate = as.numeric(est_ae[, "Estimate"]),
        se       = as.numeric(est_ae[, "SE"])
      ),
      file.path(out_dir_ae, "loo_estimates.csv")
    )

    pk_ae <- loo::pareto_k_table(loo_ae)
    suppressWarnings(
      readr::write_csv(as.data.frame(pk_ae), file.path(out_dir_ae, "pareto_k.csv"))
    )

    saveRDS(loo_ae, file.path(out_dir_ae, "loo.rds"))

  } else {
    readr::write_lines(
      "Found 'log_lik' but it contained 0 columns.",
      file.path(out_dir_ae, "loo.txt")
    )
  }

} else {
  readr::write_lines(
    "log_lik not found in fitted draws; LOO not computed.",
    file.path(out_dir_ae, "loo.txt")
  )
}

# ----------------------------------------------------------
#   SAVE FIT OBJECT
# ----------------------------------------------------------

saveRDS(
  fit_bb_tgm_ctx_stgp_ctx_attica_egypt,
  file.path(out_dir_ae, paste0(fit_name_ae, ".rds"))
)

# ----------------------------------------------------------
#   INDEX TABLES  (site / type / time — same structure as
#   the base model; region label and indicators added to
#   the site index for traceability)
# ----------------------------------------------------------

tibble::tibble(type_id = seq_along(type_levels), type = type_levels) |>
  readr::write_csv(file.path(out_dir_ae, "type_index.csv"))

sites_tbl_region |>
  dplyr::mutate(
    site_id     = dplyr::row_number(),
    attica_site = attica_site,
    egypt_site  = egypt_site
  ) |>
  readr::write_csv(file.path(out_dir_ae, "site_index.csv"))

tibble::tibble(
  t0_year       = time_info$t0,
  t_scale_years = time_info$scale_years,
  K_u           = dat_ae$K_u,
  coord_tol     = time_info$tol
) |>
  readr::write_csv(file.path(out_dir_ae, "time_config.csv"))

# ----------------------------------------------------------
#   FULL PARAMETER SUMMARY
# ----------------------------------------------------------

summ_all_ae <- fit_bb_tgm_ctx_stgp_ctx_attica_egypt$summary()
readr::write_csv(summ_all_ae, file.path(out_dir_ae, "summary_all.csv"))

# ----------------------------------------------------------
#   CORE SCALAR PARAMETERS
#   gamma_attica and gamma_egypt are added relative to the
#   base model; sigma_ctx is retained from both.
# ----------------------------------------------------------

core_vars_ae <- c(
  "alpha", "beta_date",
  "gamma_ctx", "gamma_attica", "gamma_egypt",
  "phi",
  "sigma_gp", "rho_s", "rho_t", "sigma_ctx"
)

summ_core_ae <- tryCatch(
  fit_bb_tgm_ctx_stgp_ctx_attica_egypt$summary(variables = core_vars_ae),
  error = function(e) tibble::tibble(note = conditionMessage(e))
)
readr::write_csv(summ_core_ae, file.path(out_dir_ae, "summary_core.csv"))

# ----------------------------------------------------------
#   TYPE FIXED EFFECTS  (sum-to-zero; beta_type[1..T])
# ----------------------------------------------------------

type_draws_ae <- tryCatch(
  posterior::as_draws_df(
    fit_bb_tgm_ctx_stgp_ctx_attica_egypt$draws(variables = "beta_type")
  ),
  error = function(e) NULL
)

if (!is.null(type_draws_ae)) {

  bt_cols_ae <- grep(
    "^beta_type\\[\\d+\\]$", names(type_draws_ae), value = TRUE
  )
  bt_idx_ae  <- as.integer(gsub(".*\\[(\\d+)\\]$", "\\1", bt_cols_ae))
  bt_cols_ae <- bt_cols_ae[order(bt_idx_ae)]

  summ_type_ae <- tibble::tibble(
    variable = bt_cols_ae,
    type_id  = seq_along(bt_cols_ae),
    type     = type_levels[seq_along(bt_cols_ae)],
    median   = sapply(bt_cols_ae, \(nm) stats::median(type_draws_ae[[nm]])),
    lo95     = sapply(bt_cols_ae, \(nm) stats::quantile(type_draws_ae[[nm]], 0.025)),
    hi95     = sapply(bt_cols_ae, \(nm) stats::quantile(type_draws_ae[[nm]], 0.975))
  )
  readr::write_csv(
    summ_type_ae,
    file.path(out_dir_ae, "summary_beta_type_fixed.csv")
  )

} else {
  message("beta_type not found in draws; summary_beta_type_fixed.csv not written.")
}

# ----------------------------------------------------------
#   f_site: overall alpha-level GP surface
# ----------------------------------------------------------

if (has_var(fit_bb_tgm_ctx_stgp_ctx_attica_egypt, "f_site")) {
  fit_bb_tgm_ctx_stgp_ctx_attica_egypt$summary(variables = "f_site") |>
    dplyr::mutate(
      site_id = as.integer(gsub("^f_site\\[(\\d+)\\]$", "\\1", variable))
    ) |>
    dplyr::left_join(
      sites_tbl_region |> dplyr::mutate(site_id = dplyr::row_number()),
      by = "site_id"
    ) |>
    dplyr::relocate(site_id, latitude_site, longitude_site, .after = variable) |>
    readr::write_csv(file.path(out_dir_ae, "summary_f_site.csv"))
}

# ----------------------------------------------------------
#   g_site: residual EIR-contrast GP surface
#   This is the site-level deviation from gamma_ctx *after*
#   the Attica/Egypt region terms have been accounted for.
# ----------------------------------------------------------

if (has_var(fit_bb_tgm_ctx_stgp_ctx_attica_egypt, "g_site")) {
  fit_bb_tgm_ctx_stgp_ctx_attica_egypt$summary(variables = "g_site") |>
    dplyr::mutate(
      site_id = as.integer(gsub("^g_site\\[(\\d+)\\]$", "\\1", variable))
    ) |>
    dplyr::left_join(
      sites_tbl_region |> dplyr::mutate(site_id = dplyr::row_number()),
      by = "site_id"
    ) |>
    dplyr::relocate(site_id, latitude_site, longitude_site, .after = variable) |>
    readr::write_csv(file.path(out_dir_ae, "summary_g_site.csv"))
}

# ----------------------------------------------------------
#   region_ctx_shift_out: structured Attica/Egypt component
#   of the EIR shift at each site.
#   = gamma_attica * attica_site[s] + gamma_egypt * egypt_site[s]
#   Stored in generated quantities for easy extraction.
# ----------------------------------------------------------

if (has_var(fit_bb_tgm_ctx_stgp_ctx_attica_egypt, "region_ctx_shift_out")) {
  fit_bb_tgm_ctx_stgp_ctx_attica_egypt$summary(
    variables = "region_ctx_shift_out"
  ) |>
    dplyr::mutate(
      site_id = as.integer(
        gsub("^region_ctx_shift_out\\[(\\d+)\\]$", "\\1", variable)
      )
    ) |>
    dplyr::left_join(
      sites_tbl_region |>
        dplyr::mutate(
          site_id     = dplyr::row_number(),
          attica_site = attica_site,
          egypt_site  = egypt_site
        ),
      by = "site_id"
    ) |>
    dplyr::relocate(
      site_id, latitude_site, longitude_site, region,
      attica_site, egypt_site,
      .after = variable
    ) |>
    readr::write_csv(
      file.path(out_dir_ae, "summary_region_ctx_shift.csv")
    )
}

# ----------------------------------------------------------
#   delta_site: total EIR shift per site (primary mapping
#   quantity).
#   delta_site[s] = gamma_ctx + region_ctx_shift_out[s]
#                             + g_site[s]
#   Positive  -> Attic EIR rule active at this site.
#   Near zero -> no EIR differential (Doric or Ionic).
#   Negative  -> eta preferred in EIR contexts more than avg.
# ----------------------------------------------------------

if (has_var(fit_bb_tgm_ctx_stgp_ctx_attica_egypt, "delta_site")) {
  summ_deltasite_ae <-
    fit_bb_tgm_ctx_stgp_ctx_attica_egypt$summary(variables = "delta_site") |>
    dplyr::mutate(
      site_id = as.integer(gsub("^delta_site\\[(\\d+)\\]$", "\\1", variable))
    ) |>
    dplyr::left_join(
      sites_tbl_region |>
        dplyr::mutate(
          site_id     = dplyr::row_number(),
          attica_site = attica_site,
          egypt_site  = egypt_site
        ),
      by = "site_id"
    ) |>
    dplyr::relocate(
      site_id, latitude_site, longitude_site, region,
      attica_site, egypt_site,
      .after = variable
    )
  readr::write_csv(
    summ_deltasite_ae,
    file.path(out_dir_ae, "summary_delta_site.csv")
  )
}

# ----------------------------------------------------------
#   theta_bar quantities
# ----------------------------------------------------------

for (qty_ae in c("theta_bar", "theta_bar_out", "theta_bar_in")) {
  if (has_var(fit_bb_tgm_ctx_stgp_ctx_attica_egypt, qty_ae)) {
    fit_bb_tgm_ctx_stgp_ctx_attica_egypt$summary(variables = qty_ae) |>
      dplyr::mutate(
        row = as.integer(
          gsub(paste0("^", qty_ae, "\\[(\\d+)\\]$"), "\\1", variable)
        )
      ) |>
      dplyr::relocate(row, .before = 1) |>
      readr::write_csv(
        file.path(out_dir_ae, paste0("summary_", qty_ae, ".csv"))
      )
  }
}

# ----------------------------------------------------------
#   DIAGNOSTICS
# ----------------------------------------------------------

diag_ae <- tryCatch(
  as.data.frame(
    fit_bb_tgm_ctx_stgp_ctx_attica_egypt$diagnostic_summary()
  ),
  error = function(e) NULL
)
if (!is.null(diag_ae))
  readr::write_csv(diag_ae, file.path(out_dir_ae, "diagnostic_summary.csv"))

# ----------------------------------------------------------
#   PROVENANCE
# ----------------------------------------------------------

run_meta_ae <- list(
  model_file    = stan_file_ae,
  fit_name      = fit_name_ae,
  results_dir   = normalizePath(out_dir_ae),
  N             = dat_ae$N,
  S             = dat_ae$S,
  T             = dat_ae$T,
  K_u           = dat_ae$K_u,
  n_attica      = sum(attica_site),
  n_egypt       = sum(egypt_site),
  chains        = 4L,
  iter_warmup   = 2000L,
  iter_sampling = 2000L,
  adapt_delta   = 0.95,
  seed          = 2025L
)
readr::write_file(
  jsonlite::toJSON(run_meta_ae, auto_unbox = TRUE, pretty = TRUE),
  file.path(out_dir_ae, "run_metadata.json")
)

# ----------------------------------------------------------
#   CONSOLE SUMMARY
# ----------------------------------------------------------

message("\nCore scalar parameters (incl. gamma_attica, gamma_egypt, sigma_ctx):")
print(tibble::as_tibble(summ_core_ae))

message("\nType fixed effects (sum-to-zero beta_type):")
if (exists("summ_type_ae")) {
  print(summ_type_ae)
} else {
  message("Not available.")
}

message("\nGP hyperparameters:")
print(tibble::as_tibble(
  fit_bb_tgm_ctx_stgp_ctx_attica_egypt$summary(
    variables = c("sigma_gp", "sigma_ctx", "rho_s", "rho_t")
  )
))

message("\nRegional EIR shift parameters:")
print(tibble::as_tibble(
  fit_bb_tgm_ctx_stgp_ctx_attica_egypt$summary(
    variables = c("gamma_ctx", "gamma_attica", "gamma_egypt")
  )
))

message("\nregion_ctx_shift_out (structured Attica/Egypt component; first 10 sites):")
if (has_var(fit_bb_tgm_ctx_stgp_ctx_attica_egypt, "region_ctx_shift_out")) {
  print(head(
    fit_bb_tgm_ctx_stgp_ctx_attica_egypt$summary(
      variables = "region_ctx_shift_out"
    ),
    10
  ))
} else {
  message("Not available.")
}

message("\ndelta_site (total EIR shift; first 10 sites):")
if (exists("summ_deltasite_ae")) {
  print(head(summ_deltasite_ae, 10))
} else {
  message("Not available.")
}
