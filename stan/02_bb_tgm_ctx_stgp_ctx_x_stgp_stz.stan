// Beta-binomial two-outcome model for alpha / eta variants

data {
  // ── Outcome counts per inscription ────────────────────────────────────
  int<lower=1> N;

  array[N] int<lower=0> y1;   // alpha tokens in outer context
  array[N] int<lower=0> n1;   // total tokens in outer context
  array[N] int<lower=0> y2;   // alpha tokens in inner context
  array[N] int<lower=0> n2;   // total tokens in inner context

  // ── Date-uncertainty quadrature (standardised) ─────────────────────
  int<lower=1> K_u;
  vector[K_u]  u_grid;       // K_u nodes on [0, 1]
  vector[N]    t_min_std;    // standardised lower date bound per inscription
  vector[N]    t_max_std;    // standardised upper date bound per inscription

  // ── Text type (sum-to-zero coded) ─────────────────────────────────
  int<lower=1>           T;
  array[N] int<lower=1> type;

  // ── Spatiotemporal GP: site structure ─────────────────────────────
  int<lower=1>                    S;
  array[N] int<lower=1, upper=S> site_id;
  vector[S]                       latitude_site;
  vector[S]                       longitude_site;
}

transformed data {
  // ── Standardise site spatial coordinates ──────────────────────────
  real lat_mean = mean(latitude_site);
  real lon_mean = mean(longitude_site);
  real lat_sd   = sd(latitude_site);
  real lon_sd   = sd(longitude_site);

  vector[S] x1 = (latitude_site  - lat_mean) / (lat_sd + 1e-9);
  vector[S] x2 = (longitude_site - lon_mean) / (lon_sd + 1e-9);

  // ── Per-site mean standardised mid-date (temporal GP coordinate) ───
  // Derived from inscription-level bounds; no extra data input required.
  // Intra-site temporal variation (substantial at e.g. Athens, Knidos)
  // is handled by the parametric beta_date term, not the GPs.
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

  // ── S × S squared-distance matrices (precomputed; parameters-free) ─
  // Both GPs share these matrices; rho_s and rho_t scale them inside
  // transformed parameters.
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
  real alpha;          // global intercept (log-odds of alpha)
  real beta_date;      // linear temporal trend (koine spread)

  // Text-type effects (raw; sum-to-zero centered in transformed parameters)
  vector[T] beta_type_raw;

  // Global mean inner-context shift (log-odds scale)
  real gamma_ctx;

  // Beta-binomial concentration
  real<lower=0> phi;

  // ── GP 1: overall alpha level (f_site) ──────────────────────────
  real<lower=0> sigma_gp;   // marginal SD
  real<lower=0> rho_s;      // spatial  length scale (standardised coords)
  real<lower=0> rho_t;      // temporal length scale (standardised dates)
                             //   1 unit ≈ 120 years
  vector[S] z_gp;           // non-centred weights

  // ── GP 2: inner-contrast surface (g_site) — new ───────────────────
  // Shares rho_s and rho_t with f_site (same correlation structure).
  // Independent marginal SD and weight vector.
  real<lower=0> sigma_ctx;  // marginal SD of the inner-contrast GP
  vector[S] z_ctx;          // non-centred weights (independent of z_gp)
}

transformed parameters {
  // ── Sum-to-zero centering of type effects ─────────────────────────
  vector[T] beta_type = beta_type_raw - mean(beta_type_raw);

  // ── Both GP draws from a single local Cholesky (K_site not saved) ─
  vector[S] f_site;
  vector[S] g_site;
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
    g_site = (sigma_ctx / sigma_gp) * (L_K * z_ctx);
  }
}

model {
  // ── Priors ────────────────────────────────────────────────────────
  alpha         ~ normal(0, 1);
  beta_date     ~ normal(0, 2);
  beta_type_raw ~ normal(0, 2);
  gamma_ctx     ~ normal(0, 2);
  phi           ~ lognormal(log(5), 0.6);

  // GP 1 (overall level)
  sigma_gp ~ normal(0, 1);
  rho_s    ~ lognormal(0, 0.8);   // median ≈ 1 standardised spatial unit
  rho_t    ~ lognormal(0, 0.8);   // median ≈ 120 yr; 5–95 pct: 32–446 yr
  z_gp     ~ normal(0, 1);

  // GP 2 (inner-contrast surface)
  sigma_ctx ~ normal(0, 1);       // same weakly informative prior as sigma_gp
  z_ctx     ~ normal(0, 1);

  // ── Likelihood: quadrature over date uncertainty ──────────────────
  for (i in 1:N) {
    array[K_u] real ll_terms;
    real t_lo = t_min_std[i];
    real t_hi = t_max_std[i];
    real dt   = t_hi - t_lo;

    real type_contrib = beta_type[type[i]];
    real gp_contrib   = f_site[site_id[i]];
    real ctx_contrib  = g_site[site_id[i]];   // site-specific inner deviation

    for (k in 1:K_u) {
      real t   = t_lo + u_grid[k] * dt;
      real eta = alpha + beta_date * t + type_contrib + gp_contrib;

      real mu1 = inv_logit(eta);                              // outer
      real a1  = mu1 * phi;
      real b1  = (1.0 - mu1) * phi;

      real mu2 = inv_logit(eta + gamma_ctx + ctx_contrib);    // inner
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

  // Total inner shift per site on the log-odds scale.
  // This is the primary quantity for spatial mapping: positive values
  // indicate that inner context favour alpha (Attic-like), near-zero
  // indicates no inner differential (Doric: alpha everywhere; Ionic: eta
  // everywhere), negative indicates eta preferred in inner context.
  vector[S] delta_site;

  int mid_idx = (K_u + 1) %/% 2;

  // Compute delta_site from the posterior draw of gamma_ctx and g_site.
  for (s in 1:S) {
    delta_site[s] = gamma_ctx + g_site[s];
  }

  for (i in 1:N) {
    array[K_u] real ll_terms;
    real t_lo = t_min_std[i];
    real t_hi = t_max_std[i];
    real dt   = t_hi - t_lo;

    real mu_sum_out = 0.0;
    real mu_sum_in  = 0.0;

    real type_contrib = beta_type[type[i]];
    real gp_contrib   = f_site[site_id[i]];
    real ctx_contrib  = g_site[site_id[i]];

    for (k in 1:K_u) {
      real t   = t_lo + u_grid[k] * dt;
      real eta = alpha + beta_date * t + type_contrib + gp_contrib;

      real mu1 = inv_logit(eta);
      real mu2 = inv_logit(eta + gamma_ctx + ctx_contrib);

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

    // Posterior-predictive replications using the quadrature mid-point date
    {
      real t_mid_i = t_lo + u_grid[mid_idx] * dt;
      real eta_mid = alpha + beta_date * t_mid_i + type_contrib + gp_contrib;

      real mu1_mid = inv_logit(eta_mid);
      real mu2_mid = inv_logit(eta_mid + gamma_ctx + ctx_contrib);

      y1_rep[i] = beta_binomial_rng(n1[i],
                    mu1_mid * phi, (1.0 - mu1_mid) * phi);
      y2_rep[i] = beta_binomial_rng(n2[i],
                    mu2_mid * phi, (1.0 - mu2_mid) * phi);
    }
  }
}
