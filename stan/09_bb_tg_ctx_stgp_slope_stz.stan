data {
  // ── Outcome counts per inscription ──────────────────────────────────
  int<lower=1> N;

  array[N] int<lower=0> y1;   // alpha tokens in outer context
  array[N] int<lower=0> n1;   // total tokens in outer context
  array[N] int<lower=0> y2;   // alpha tokens in inner context
  array[N] int<lower=0> n2;   // total tokens in inner context

  // ── Date-uncertainty quadrature (standardised) ──────────────────────
  int<lower=1> K_u;
  vector[K_u]  u_grid;       // K_u nodes on (0, 1)
  vector[N]    t_min_std;    // standardised lower date bound per inscription
  vector[N]    t_max_std;    // standardised upper date bound per inscription

  // ── Text type (sum-to-zero coded) ───────────────────────────────────
  int<lower=1>                    T;
  array[N] int<lower=1, upper=T> type;

  // ── Spatiotemporal GP: site structure ───────────────────────────────
  int<lower=1>                    S;
  array[N] int<lower=1, upper=S> site_id;
  vector[S]                       latitude_site;
  vector[S]                       longitude_site;
}

transformed data {
  // ── Standardise site spatial coordinates ────────────────────────────
  real lat_mean = mean(latitude_site);
  real lon_mean = mean(longitude_site);
  real lat_sd   = sd(latitude_site);
  real lon_sd   = sd(longitude_site);

  vector[S] x1 = (latitude_site  - lat_mean) / (lat_sd + 1e-9);
  vector[S] x2 = (longitude_site - lon_mean) / (lon_sd + 1e-9);

  // ── Per-site mean standardised mid-date (temporal GP coordinate) ────
  vector[S] t_site;
  {
    vector[S] cnt = rep_vector(0.0, S);
    t_site = rep_vector(0.0, S);
    for (i in 1:N) {
      real tmid = (t_min_std[i] + t_max_std[i]) * 0.5;
      t_site[site_id[i]] += tmid;
      cnt[site_id[i]]    += 1.0;
    }
    for (s in 1:S) {
      t_site[s] /= cnt[s];
    }
  }

  // ── S × S squared-distance matrices (precomputed; parameters-free) ──
  matrix[S, S] sq_dist_space;
  matrix[S, S] sq_dist_time;

  for (i in 1:S) {
    sq_dist_space[i, i] = 0.0;
    sq_dist_time[i, i]  = 0.0;
    for (j in (i + 1):S) {
      real dx1 = x1[i] - x1[j];
      real dx2 = x2[i] - x2[j];
      real ds2 = dx1 * dx1 + dx2 * dx2;
      real dt2 = square(t_site[i] - t_site[j]);

      sq_dist_space[i, j] = ds2;
      sq_dist_space[j, i] = ds2;
      sq_dist_time[i, j]  = dt2;
      sq_dist_time[j, i]  = dt2;
    }
  }
}

parameters {
  // ── Shared parameters (identical to Model 1) ────────────────────────
  real alpha;       // global intercept (log-odds of alpha, outer)
  real beta_date;   // global temporal trend (koine spread)

  vector[T] beta_type_raw;   // text-type effects (raw; sum-to-zero centred)

  real           gamma_ctx;  // scalar inner context shift
  real<lower=0>  phi;        // beta-binomial concentration

  // Spatiotemporal GP hyperparameters (for f_site)
  real<lower=0> sigma_gp;   // marginal SD
  real<lower=0> rho_s;      // spatial  length scale (standardised coords)
  real<lower=0> rho_t;      // temporal length scale (1 unit ≈ 120 years)

  vector[S] z_gp;            // non-centred weights for f_site

  // ── Slope GP parameters ────────────────────────────────────────
  real<lower=0> sigma_slope; // marginal SD of the slope deviation surface
  real<lower=0> rho_slope;   // spatial length scale of the slope GP
                              // (same units as rho_s: standardised coords)

  vector[S] z_slope;         // non-centred weights for b_site
}

transformed parameters {
  // ── Sum-to-zero centring of type effects ────────────────────────────
  vector[T] beta_type = beta_type_raw - mean(beta_type_raw);

  // ── Spatiotemporal GP: f_site (intercept surface) ───────────────────
  vector[S] f_site;
  {
    real sq_sigma   = square(sigma_gp);
    real inv_2rhos2 = 0.5 / square(rho_s);
    real inv_2rhot2 = 0.5 / square(rho_t);

    matrix[S, S] K_site;
    for (i in 1:S) {
      for (j in i:S) {
        real kij = sq_sigma
                 * exp(- inv_2rhos2 * sq_dist_space[i, j]
                       - inv_2rhot2 * sq_dist_time[i, j]);
        K_site[i, j] = kij;
        K_site[j, i] = kij;
      }
      K_site[i, i] += 1e-6;
    }
    matrix[S, S] L_K = cholesky_decompose(K_site);
    f_site = L_K * z_gp;
  }

  // ── Spatial GP: b_site (slope deviation surface) ────────────────────
  vector[S] b_site;
  {
    real sq_sigma_slope = square(sigma_slope);
    real inv_2rho2_sl   = 0.5 / square(rho_slope);

    matrix[S, S] K_slope;
    for (i in 1:S) {
      for (j in i:S) {
        real kij = sq_sigma_slope
                 * exp(- inv_2rho2_sl * sq_dist_space[i, j]);
        K_slope[i, j] = kij;
        K_slope[j, i] = kij;
      }
      K_slope[i, i] += 1e-6;
    }
    matrix[S, S] L_K_slope = cholesky_decompose(K_slope);
    b_site = L_K_slope * z_slope;
  }
}

model {
  // ── Priors: shared parameters (identical to Model 1) ────────────────
  alpha         ~ normal(0, 1);
  beta_date     ~ normal(0, 2);
  beta_type_raw ~ normal(0, 2);
  gamma_ctx     ~ normal(0, 2);
  phi           ~ lognormal(log(5), 0.6);

  sigma_gp ~ normal(0, 1);
  rho_s    ~ lognormal(0, 0.8);   // median ≈ 1 standardised spatial unit
  rho_t    ~ lognormal(0, 0.8);   // median ≈ 120 yr; 5–95 pct: 32–446 yr
  z_gp     ~ normal(0, 1);

  // ── Priors: slope GP ───────────────────────────────────────────
  sigma_slope ~ normal(0, 1);
  rho_slope   ~ lognormal(0, 0.8);
  z_slope     ~ normal(0, 1);

  // ── Likelihood: quadrature over date uncertainty ─────────────────────
  for (i in 1:N) {
    array[K_u] real ll_terms;
    real t_lo = t_min_std[i];
    real t_hi = t_max_std[i];
    real dt   = t_hi - t_lo;

    real type_contrib = beta_type[type[i]];
    real gp_contrib   = f_site[site_id[i]];
    real slope_i      = beta_date + b_site[site_id[i]];

    for (k in 1:K_u) {
      real t   = t_lo + u_grid[k] * dt;
      real eta = alpha + slope_i * t + type_contrib + gp_contrib;

      real mu1 = inv_logit(eta);
      real a1  = mu1 * phi;
      real b1  = (1.0 - mu1) * phi;

      real mu2 = inv_logit(eta + gamma_ctx);
      real a2  = mu2 * phi;
      real b2  = (1.0 - mu2) * phi;

      ll_terms[k] =
            lchoose(n1[i], y1[i])
          + lbeta(a1 + y1[i], b1 + n1[i] - y1[i]) - lbeta(a1, b1)
          + lchoose(n2[i], y2[i])
          + lbeta(a2 + y2[i], b2 + n2[i] - y2[i]) - lbeta(a2, b2);
    }

    target += log_sum_exp(ll_terms) - log(K_u);
  }
}

generated quantities {
  vector[N]                    log_lik;
  array[N] int                 y1_rep;
  array[N] int                 y2_rep;
  vector<lower=0, upper=1>[N] theta_bar_out;   // mean P(alpha) in outer
  vector<lower=0, upper=1>[N] theta_bar_in;    // mean P(alpha) in inner
  vector<lower=0, upper=1>[N] theta_bar;       // n-weighted average

  // Site-level effective temporal trend: beta_date + b_site[s].
  vector[S] beta_date_site;
  for (s in 1:S) {
    beta_date_site[s] = beta_date + b_site[s];
  }

  int mid_idx = (K_u + 1) %/% 2;

  for (i in 1:N) {
    array[K_u] real ll_terms;
    real t_lo = t_min_std[i];
    real t_hi = t_max_std[i];
    real dt   = t_hi - t_lo;

    real mu_sum_out = 0.0;
    real mu_sum_in  = 0.0;

    real type_contrib = beta_type[type[i]];
    real gp_contrib   = f_site[site_id[i]];
    real slope_i      = beta_date + b_site[site_id[i]];

    for (k in 1:K_u) {
      real t   = t_lo + u_grid[k] * dt;
      real eta = alpha + slope_i * t + type_contrib + gp_contrib;

      real mu1 = inv_logit(eta);
      real mu2 = inv_logit(eta + gamma_ctx);

      real a1 = mu1 * phi;
      real b1 = (1.0 - mu1) * phi;
      real a2 = mu2 * phi;
      real b2 = (1.0 - mu2) * phi;

      ll_terms[k] =
            lchoose(n1[i], y1[i])
          + lbeta(a1 + y1[i], b1 + n1[i] - y1[i]) - lbeta(a1, b1)
          + lchoose(n2[i], y2[i])
          + lbeta(a2 + y2[i], b2 + n2[i] - y2[i]) - lbeta(a2, b2);

      mu_sum_out += mu1;
      mu_sum_in  += mu2;
    }

    log_lik[i]       = log_sum_exp(ll_terms) - log(K_u);
    theta_bar_out[i] = mu_sum_out / K_u;
    theta_bar_in[i]  = mu_sum_in  / K_u;

    {
      real denom = n1[i] + n2[i];
      theta_bar[i] = (denom > 0)
                   ? (n1[i] * theta_bar_out[i] + n2[i] * theta_bar_in[i]) / denom
                   : 0.5;
    }

    // Posterior-predictive replications at the quadrature mid-point date
    {
      real t_mid_i = t_lo + u_grid[mid_idx] * dt;
      real eta_mid = alpha + slope_i * t_mid_i + type_contrib + gp_contrib;

      real mu1_mid = inv_logit(eta_mid);
      real mu2_mid = inv_logit(eta_mid + gamma_ctx);

      y1_rep[i] = beta_binomial_rng(n1[i],
                    mu1_mid * phi, (1.0 - mu1_mid) * phi);
      y2_rep[i] = beta_binomial_rng(n2[i],
                    mu2_mid * phi, (1.0 - mu2_mid) * phi);
    }
  }
}
