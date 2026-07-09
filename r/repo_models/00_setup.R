# ==========================================================
#  00_setup.R
#
#  Shared infrastructure for all model-fitting scripts.
#
#  Provides:
#    • Package loading
#    • results_root path
#    • Utility functions: %||%, has_var, get_chain_id_safe,
#      standardize_dates, ensure_std_time, get_agg_data
#    • Global time constants: t0_year, t_scale_years, K_u
#    • Aggregated datasets: agg_data, agg_data_dialect,
#      agg_data_basic_raw, agg_data_dialect_raw
#    • Data builder: build_data_stgp_ctx()
#    • Shared Stan data: dat_stgp_ctx, type_levels,
#      sites_tbl, time_info
# ==========================================================

# ----------------------------------------------------------
#  PACKAGES
# ----------------------------------------------------------

library(posterior)
library(dplyr)
library(tidyr)
library(stringr)
library(forcats)
library(ggplot2)
library(scales)
library(purrr)
library(cmdstanr)
library(readr)
library(readxl)
library(loo)
library(jsonlite)
library(tibble)
library(here)

options(dplyr.summarise.inform = FALSE)

# ----------------------------------------------------------
#  PATHS
# ----------------------------------------------------------

paths <- list(
  data = here::here("data"),
  stan = here::here("stan"),
  results = here::here("results"),
  r = here::here("r")
)

results_root <- here::here("results")

# ----------------------------------------------------------
#  UTILITY FUNCTIONS
# ----------------------------------------------------------

`%||%` <- function(a, b) if (!is.null(a)) a else b

# has_var: check whether a Stan variable name is present in a
# fitted CmdStanR model object.  Tries metadata first; falls
# back to a single-element draw attempt.
has_var <- function(fit, v) {
  md <- tryCatch(fit$metadata(), error = function(e) NULL)
  if (!is.null(md) && !is.null(md$stan_variables)) {
    return(any(grepl(paste0("^", v, "(\\[|$)"), md$stan_variables)))
  }
  tryCatch({
    suppressMessages(fit$draws(variables = paste0(v, "[1]"), format = "matrix"))
    TRUE
  }, error = function(e) FALSE)
}

# get_chain_id_safe: reconstruct per-draw chain IDs from a
# fitted model and its log-likelihood matrix.
get_chain_id_safe <- function(fit, ll_mat) {
  md <- tryCatch(fit$metadata(), error = function(e) list())
  chains_md <- suppressWarnings(as.integer(md$chains %||% md$num_chains))
  iters_md  <- suppressWarnings(as.integer(md$iter_sampling))

  ok_meta <- is.finite(chains_md) && chains_md > 0L &&
             is.finite(iters_md)  && iters_md  > 0L &&
             chains_md * iters_md == nrow(ll_mat)
  if (ok_meta) return(rep(seq_len(chains_md), each = iters_md))

  dr <- tryCatch(
    fit$draws(variables = "log_lik", format = "draws_array"),
    error = function(e) NULL
  )
  if (!is.null(dr)) {
    n_ch <- posterior::nchains(dr)
    n_it <- posterior::ndraws(dr) / n_ch
    if (n_ch > 0 && n_it > 0 && isTRUE(all.equal(n_ch * n_it, nrow(ll_mat))))
      return(rep(seq_len(n_ch), each = n_it))
  }

  rep(1L, nrow(ll_mat))
}

# ----------------------------------------------------------
#  TIME UTILITIES
# ----------------------------------------------------------

standardize_dates <- function(df, t0 = 0, scale_years = 100) {
  stopifnot(all(c("date_min", "date_max") %in% names(df)))
  out <- df %>%
    dplyr::mutate(
      date_min  = as.numeric(.data$date_min),
      date_max  = as.numeric(.data$date_max),
      t_min_std = (pmin(.data$date_min, .data$date_max) - t0) / scale_years,
      t_max_std = (pmax(.data$date_min, .data$date_max) - t0) / scale_years
    )
  attr(out, ".time") <- list(t0 = t0, scale_years = scale_years)
  out
}

ensure_std_time <- function(df, t0, scale_years) {
  if (all(c("t_min_std", "t_max_std") %in% names(df))) {
    tm        <- attr(df, ".time")
    t0_old    <- tm$t0 %||% NA_real_
    scale_old <- tm$scale_years %||% NA_real_
    if (!isTRUE(all.equal(t0_old, t0)) ||
        !isTRUE(all.equal(scale_old, scale_years))) {
      warning(sprintf(
        "Data already standardised with t0=%s, scale=%s; ",
        as.character(t0_old), as.character(scale_old)
      ), sprintf(
        "requested t0=%s, scale=%s. Re-using existing columns.",
        as.character(t0), as.character(scale_years)
      ))
    }
    return(df)
  }
  standardize_dates(df, t0 = t0, scale_years = scale_years)
}

# ----------------------------------------------------------
#  LOAD AGGREGATED DATA
# ----------------------------------------------------------

source(here::here("r", "repo_agg_data.R"))

# Primary aggregated data (dialect columns required for models
# that use them; keep_one_sided = TRUE retains texts with
# tokens in only one context).
agg_data <- make_agg_data(
  exclude_types  = "erotic",
  need_dialect   = TRUE,
  keep_one_sided = TRUE
)

agg_data <- agg_data %>%
  mutate(
    has_n1   = n1 > 0,
    has_n2   = n2 > 0,
    has_both = has_n1 & has_n2
  )

stopifnot(!"erotic" %in% unique(agg_data$type))

# Dialect-aware dataset (same rows, same order; required by
# models 05 and 09; NULL if dialect columns are unavailable).
agg_data_dialect <- tryCatch(
  make_agg_data(exclude_types = "erotic", need_dialect = TRUE),
  error = function(e) {
    message("Dialect columns not available: ", e$message)
    NULL
  }
)

# Keep raw (unstandardised) versions for the time-origin
# calculation below and for passing to build_data_stgp_ctx().
agg_data_basic_raw   <- agg_data
agg_data_dialect_raw <- agg_data_dialect

# ----------------------------------------------------------
#  TIME CONSTANTS
# ----------------------------------------------------------

# t0_year: median midpoint date across all inscriptions, used
# as the time origin for standardisation.  This value is data-
# driven and must be computed before any model scripts run.
t0_year       <- with(agg_data_basic_raw,
                      median((date_min + date_max) / 2, na.rm = TRUE))
t_scale_years <- 100   # 1 standardised time unit = 100 years
K_u           <- 11L   # number of midpoint nodes

# Standardised datasets
agg_data_basic   <- ensure_std_time(
  agg_data_basic_raw, t0 = t0_year, scale_years = t_scale_years
)
agg_data_dialect <- if (!is.null(agg_data_dialect_raw)) {
  ensure_std_time(
    agg_data_dialect_raw, t0 = t0_year, scale_years = t_scale_years
  )
} else {
  NULL
}

# Convenience accessor: returns the correct standardised dataset
# depending on whether dialect columns are required.
get_agg_data <- function(require_dialect = FALSE) {
  if (require_dialect) {
    if (is.null(agg_data_dialect))
      stop("This model requires dialect/dialect_group columns, ",
           "but none were found.")
    return(agg_data_dialect)
  }
  agg_data_basic
}

# ----------------------------------------------------------
#  DATA BUILDER — STGP models
#
#  Builds the Stan data list used by models 01, 02, 03, 06,
#  07, 08, 09, and 10.  Models 04 and 05 use their own
#  builders defined within their respective scripts.
# ----------------------------------------------------------

build_data_stgp_ctx <- function(
  df,
  K_u         = 11L,
  t0          = 0,
  scale_years = 100,
  tol         = 1e-5
) {
  need <- c("y1", "n1", "y2", "n2", "date_min", "date_max",
            "type", "latitude", "longitude")
  miss <- setdiff(need, names(df))
  if (length(miss)) stop("Missing columns: ", paste(miss, collapse = ", "))

  df_s <- ensure_std_time(df, t0 = t0, scale_years = scale_years)

  if (is.factor(df_s$type)) {
    type_levels <- levels(df_s$type)
  } else {
    df_s$type   <- trimws(as.character(df_s$type))
    type_levels <- sort(unique(df_s$type))
    if ("funerary" %in% type_levels)
      type_levels <- c("funerary", setdiff(type_levels, "funerary"))
  }
  type_int <- as.integer(factor(as.character(df_s$type),
                                levels = type_levels))

  df_s <- df_s %>%
    dplyr::mutate(
      latitude  = as.numeric(latitude),
      longitude = as.numeric(longitude)
    )
  if (any(!is.finite(df_s$latitude)) || any(!is.finite(df_s$longitude)))
    stop("Non-finite latitude/longitude values found.")

  round_to <- function(x, tol) round(x / tol) * tol
  df_s <- df_s %>%
    dplyr::mutate(
      lat_k = round_to(latitude,  tol),
      lon_k = round_to(longitude, tol)
    )

  sites_lut <- df_s %>%
    dplyr::distinct(lat_k, lon_k) %>%
    dplyr::arrange(lat_k, lon_k) %>%
    dplyr::mutate(site = dplyr::row_number())

  df_s <- df_s %>%
    dplyr::left_join(sites_lut, by = c("lat_k", "lon_k"))

  S <- nrow(sites_lut)

  sites_tbl <- df_s %>%
    dplyr::group_by(site) %>%
    dplyr::summarise(
      latitude_site  = mean(latitude,  na.rm = TRUE),
      longitude_site = mean(longitude, na.rm = TRUE),
      .groups        = "drop"
    ) %>%
    dplyr::arrange(site)

  stopifnot(nrow(sites_tbl) == S)

  u_grid <- (seq_len(K_u) - 0.5) / K_u

  data_list <- list(
    N              = nrow(df_s),
    y1             = as.integer(df_s$y1),
    n1             = as.integer(df_s$n1),
    y2             = as.integer(df_s$y2),
    n2             = as.integer(df_s$n2),
    K_u            = as.integer(K_u),
    u_grid         = as.numeric(u_grid),
    t_min_std      = as.numeric(df_s$t_min_std),
    t_max_std      = as.numeric(df_s$t_max_std),
    T              = length(type_levels),
    type           = type_int,
    S              = S,
    site_id        = as.integer(df_s$site),
    latitude_site  = as.numeric(sites_tbl$latitude_site),
    longitude_site = as.numeric(sites_tbl$longitude_site)
  )

  stopifnot(all(data_list$type    >= 1 & data_list$type    <= data_list$T))
  stopifnot(all(data_list$site_id >= 1 & data_list$site_id <= data_list$S))
  stopifnot(
    lengths(data_list[c("y1", "n1", "y2", "n2",
                        "t_min_std", "t_max_std",
                        "type", "site_id")]) == data_list$N
  )
  stopifnot(length(data_list$latitude_site)  == data_list$S)
  stopifnot(length(data_list$longitude_site) == data_list$S)
  stopifnot(length(data_list$u_grid)         == data_list$K_u)

  attr(data_list, ".levels") <- list(type = type_levels)
  attr(data_list, ".sites")  <- sites_tbl
  attr(data_list, ".time")   <- list(t0 = t0, scale_years = scale_years,
                                     tol = tol)
  data_list
}

# ----------------------------------------------------------
#  BUILD SHARED STAN DATA
# ----------------------------------------------------------

dat_stgp_ctx <- build_data_stgp_ctx(
  df          = get_agg_data(require_dialect = FALSE),
  K_u         = K_u,
  t0          = t0_year,
  scale_years = t_scale_years,
  tol         = 1e-5
)

type_levels <- attr(dat_stgp_ctx, ".levels")$type
sites_tbl   <- attr(dat_stgp_ctx, ".sites")
time_info   <- attr(dat_stgp_ctx, ".time")

message(sprintf(
  "Setup complete: N=%d inscriptions, S=%d sites, T=%d types, K_u=%d nodes.",
  dat_stgp_ctx$N, dat_stgp_ctx$S, dat_stgp_ctx$T, dat_stgp_ctx$K_u
))
