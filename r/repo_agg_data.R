make_agg_data <- function(
    data_path = here::here("data", "data_2026_06_30.csv"),
    exclude_types = c("erotic"),
    need_dialect  = FALSE,
    dialect_group_candidates = c("dialect_group", "dia_group", "group", "dialectGroup"),
    dialect_candidates       = c("dialect", "dialect_name", "dia", "variety"),
    keep_one_sided = TRUE
) {
  library(dplyr); library(readr); library(tidyr)

  .find_col <- function(nm, cand, data) {
    hit <- cand[cand %in% names(data)]
    if (length(hit)) hit[1] else NA_character_
  }
  .norm_chr <- function(x) trimws(tolower(as.character(x)))

  col_alpha_eta <- "outcome"
  col_eir       <- "preceding_eir"
  col_dmin      <- "date_min"
  col_dmax      <- "date_max"
  col_lat       <- "latitude"
  col_lon       <- "longitude"

  raw <- readr::read_csv(data_path, show_col_types = FALSE) %>%
    mutate(
      region = trimws(region),
      region = dplyr::recode(
        region,
        "Rhodos Isl."   = "Rhodes",
        "Rhodes Isl."   = "Rhodes",
        "Rhodos Island" = "Rhodes",
        .default        = region
      ),
      type_raw = trimws(type),
      type_std = tolower(type_raw)
    ) %>%
    tidyr::drop_na(dplyr::all_of(c(col_lat, col_lon)))

  dg_col <- .find_col("dialect_group", dialect_group_candidates, raw)
  d_col  <- .find_col("dialect",       dialect_candidates,       raw)

  if (need_dialect && (is.na(dg_col) || is.na(d_col))) {
    stop("Dialect columns requested but not found.")
  }

  raw <- raw %>% filter(!(tolower(type_std) %in% tolower(exclude_types)))

  records <- raw %>%
    transmute(
      text,
      type       = .norm_chr(type_std),
      region     = region,
      outcome1   = suppressWarnings(as.integer(.data[[col_alpha_eta]])),
      n_eir      = suppressWarnings(as.integer(.data[[col_eir]])),
      date_min   = suppressWarnings(as.numeric(.data[[col_dmin]])),
      date_max   = suppressWarnings(as.numeric(.data[[col_dmax]])),
      latitude   = suppressWarnings(as.numeric(.data[[col_lat]])),
      longitude  = suppressWarnings(as.numeric(.data[[col_lon]])),
      dialect_group = if (!is.na(dg_col)) .norm_chr(.data[[dg_col]]) else NA_character_,
      dialect       = if (!is.na(d_col))  .norm_chr(.data[[d_col]])  else NA_character_
    ) %>%
    filter(is.finite(latitude), is.finite(longitude)) %>%
    mutate(
      dialect_group = dplyr::case_when(
        is.na(dialect_group) ~ NA_character_,
        dialect_group %in% c("doric", "west greek") ~ "doric",
        TRUE ~ dialect_group
      ),
      dialect = dplyr::case_when(
        dialect == "east aeolic"    ~ "lesbian",
        dialect == "koine"          ~ "attic",
        dialect == "west aegean"    ~ "melian",
        dialect == "east aegean"    ~ "rhodian",
        dialect == "east cretan"    ~ "cretan",
        dialect == "central cretan" ~ "cretan",
        dialect == "west cretan"    ~ "cretan",
        TRUE ~ dialect
      )
    )

  agg_data <- records %>%
    filter(!is.na(outcome1), !is.na(n_eir),
           is.finite(date_min), is.finite(date_max)) %>%
    group_by(text) %>%
    summarise(
      y1            = sum(outcome1 == 1 & n_eir == 0, na.rm = TRUE),
      n1            = sum(n_eir == 0,                  na.rm = TRUE),
      y2            = sum(outcome1 == 1 & n_eir == 1, na.rm = TRUE),
      n2            = sum(n_eir == 1,                  na.rm = TRUE),
      dmin_mean     = mean(date_min,  na.rm = TRUE),
      dmax_mean     = mean(date_max,  na.rm = TRUE),
      type          = first(type),
      region        = first(region),
      latitude      = mean(latitude,  na.rm = TRUE),
      longitude     = mean(longitude, na.rm = TRUE),
      dialect_group = first(dialect_group),
      dialect       = first(dialect),
      .groups = "drop"
    ) %>%
    mutate(
      date_min = pmin(dmin_mean, dmax_mean, na.rm = TRUE),
      date_max = pmax(dmin_mean, dmax_mean, na.rm = TRUE),
      type     = ifelse(is.na(type) | type == "", "unknown", type)
    ) %>%
    {
      sel <- c("text", "y1", "n1", "y2", "n2", "date_min", "date_max",
               "type", "region", "latitude", "longitude")
      if (!is.na(dg_col) && !is.na(d_col))
        sel <- c(sel, "dialect_group", "dialect")
      dplyr::select(., dplyr::all_of(sel))
    } %>%
    {
      if (keep_one_sided) dplyr::filter(., (n1 + n2) > 0)
      else                dplyr::filter(., n1 > 0 & n2 > 0)
    }

  # TYPE: funerary as baseline if present
  type_levels <- sort(unique(agg_data$type))
  if ("funerary" %in% type_levels) {
    type_levels <- c("funerary", setdiff(type_levels, "funerary"))
  }
  agg_data$type <- factor(agg_data$type, levels = type_levels)

  agg_data
}
