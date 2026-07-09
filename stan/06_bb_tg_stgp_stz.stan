// Beta-binomial no-context model for alpha / eta variants

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
  // ── Pool inner and outer tokens ─────────────────────────────────────
  // The no-context model treats all tokens within an inscription as
  // exchangeable: there is no distinction between inner and outer
  // contexts.  The pooled counts form the single outcome.
  // Terms where n_total[i] = 0 and y_total[i] = 0 contribute exactly 0
  // to the log-likelihood (lchoose(0,0) = 0; lbeta cancels), so no
  // guard is needed.
  array[N] int<lower=0> y_total;
  array[N] int<lower=0> n_total;
  for (i in 1:N) {
    y_total[i] = y1[i] + y2[i];
    n_total[i] = n1[i] + n2[i];
  }

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
  real alpha;       // global intercept (log-odds of alpha)
  real beta_date;   // linear temporal trend (koine spread)

  // Text-type effects (raw; sum-to-zero centred in transformed parameters)
  vector[T] beta_type_raw;

  // Beta-binomial concentration
  real<lower=0> phi;

  // Spatiotemporal GP hyperparameters
  real<lower=0> sigma_gp;   // marginal SD
  real<lower=0> rho_s;      // spatial  length scale (standardised coords)
  real<lower=0> rho_t;      // temporal length scale (standardised dates;
                             //   1 unit ≈ 120 years)

  // Non-centred GP weights
  vector[S] z_gp;
}

transformed parameters {
  // ── Sum-to-zero centring of type effects ────────────────────────────
  vector[T] beta_type = beta_type_raw - mean(beta_type_raw);

  // ── GP draw: K_site and L_K are local temporaries (not saved) ───────
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
}

model {
  // ── Priors ───────────────────────────────────────────────────────────
  alpha         ~ normal(0, 1);
  beta_date     ~ normal(0, 2);
  beta_type_raw ~ normal(0, 2);
  phi           ~ lognormal(log(5), 0.6);

  sigma_gp ~ normal(0, 1);
  rho_s    ~ lognormal(0, 0.8);   // median ≈ 1 standardised spatial unit
  rho_t    ~ lognormal(0, 0.8);   // median ≈ 120 yr; 5–95 pct: 32–446 yr
  z_gp     ~ normal(0, 1);

  // ── Likelihood: quadrature over date uncertainty ─────────────────────
  // Single pooled beta-binomial per inscription.
  for (i in 1:N) {
    array[K_u] real ll_terms;
    real t_lo = t_min_std[i];
    real t_hi = t_max_std[i];
    real dt   = t_hi - t_lo;

    real type_contrib = beta_type[type[i]];
    real gp_contrib   = f_site[site_id[i]];

    for (k in 1:K_u) {
      real t   = t_lo + u_grid[k] * dt;
      real eta = alpha + beta_date * t + type_contrib + gp_contrib;

      real mu = inv_logit(eta);
      real a  = mu * phi;
      real b  = (1.0 - mu) * phi;

      ll_terms[k] =
            lchoose(n_total[i], y_total[i])
          + lbeta(a + y_total[i], b + n_total[i] - y_total[i]) - lbeta(a, b);
    }

    target += log_sum_exp(ll_terms) - log(K_u);
  }
}

generated quantities {
  vector[N]                    log_lik;
  array[N] int                 y_rep;        // replication of pooled count
  vector<lower=0, upper=1>[N] theta_bar;    // quadrature-averaged P(alpha)

  int mid_idx = (K_u + 1) %/% 2;

  for (i in 1:N) {
    array[K_u] real ll_terms;
    real t_lo = t_min_std[i];
    real t_hi = t_max_std[i];
    real dt   = t_hi - t_lo;

    real mu_sum = 0.0;

    real type_contrib = beta_type[type[i]];
    real gp_contrib   = f_site[site_id[i]];

    for (k in 1:K_u) {
      real t   = t_lo + u_grid[k] * dt;
      real eta = alpha + beta_date * t + type_contrib + gp_contrib;

      real mu = inv_logit(eta);
      real a  = mu * phi;
      real b  = (1.0 - mu) * phi;

      ll_terms[k] =
            lchoose(n_total[i], y_total[i])
          + lbeta(a + y_total[i], b + n_total[i] - y_total[i]) - lbeta(a, b);

      mu_sum += mu;
    }

    log_lik[i]   = log_sum_exp(ll_terms) - log(K_u);
    theta_bar[i] = mu_sum / K_u;

    // Posterior-predictive replication at the quadrature mid-point date
    {
      real t_mid_i = t_lo + u_grid[mid_idx] * dt;
      real eta_mid = alpha + beta_date * t_mid_i + type_contrib + gp_contrib;
      real mu_mid  = inv_logit(eta_mid);

      y_rep[i] = beta_binomial_rng(n_total[i],
                   mu_mid * phi, (1.0 - mu_mid) * phi);
    }
  }
}
