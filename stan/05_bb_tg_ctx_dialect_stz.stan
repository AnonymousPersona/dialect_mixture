// Beta-binomial two-outcome model for alpha / eta variants

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

  // ── Dialect group (sum-to-zero coded) ───────────────────────────────
  // Integer codes 1..D assigned by the R data builder.
  // All dialect_group values not in {attic, doric, ionic} are collapsed
  // to "other" before encoding.
  int<lower=1>                    D;
  array[N] int<lower=1, upper=D> dialect;
}

parameters {
  real alpha;       // global intercept (log-odds of alpha, outer)
  real beta_date;   // linear temporal trend (koine spread)

  // Text-type effects (raw; sum-to-zero centred in transformed parameters)
  vector[T] beta_type_raw;

  // Dialect-group effects (raw; sum-to-zero centred in transformed parameters)
  vector[D] beta_dialect_raw;

  // inner-context shift
  real gamma_ctx;

  // Beta-binomial concentration
  real<lower=0> phi;
}

transformed parameters {
  // ── Sum-to-zero centring ────────────────────────────────────────────
  // Identical treatment for both type and dialect: subtract the
  // sample mean so that the effects are identified relative to the
  // grand mean captured by alpha.
  vector[T] beta_type    = beta_type_raw    - mean(beta_type_raw);
  vector[D] beta_dialect = beta_dialect_raw - mean(beta_dialect_raw);
}

model {
  // ── Priors ──────────────────────────────────────────────────────────
  alpha            ~ normal(0, 1);
  beta_date        ~ normal(0, 2);
  beta_type_raw    ~ normal(0, 2);
  beta_dialect_raw ~ normal(0, 2);
  gamma_ctx        ~ normal(0, 2);
  phi              ~ lognormal(log(5), 0.6);

  // ── Likelihood: quadrature over date uncertainty ────────────────────
  for (i in 1:N) {
    array[K_u] real ll_terms;
    real t_lo = t_min_std[i];
    real t_hi = t_max_std[i];
    real dt   = t_hi - t_lo;

    real type_contrib    = beta_type[type[i]];
    real dialect_contrib = beta_dialect[dialect[i]];

    for (k in 1:K_u) {
      real t   = t_lo + u_grid[k] * dt;
      real eta = alpha + beta_date * t + type_contrib + dialect_contrib;

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

  int mid_idx = (K_u + 1) %/% 2;

  for (i in 1:N) {
    array[K_u] real ll_terms;
    real t_lo = t_min_std[i];
    real t_hi = t_max_std[i];
    real dt   = t_hi - t_lo;

    real mu_sum_out = 0.0;
    real mu_sum_in  = 0.0;

    real type_contrib    = beta_type[type[i]];
    real dialect_contrib = beta_dialect[dialect[i]];

    for (k in 1:K_u) {
      real t   = t_lo + u_grid[k] * dt;
      real eta = alpha + beta_date * t + type_contrib + dialect_contrib;

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
      real eta_mid = alpha + beta_date * t_mid_i + type_contrib + dialect_contrib;

      real mu1_mid = inv_logit(eta_mid);
      real mu2_mid = inv_logit(eta_mid + gamma_ctx);

      y1_rep[i] = beta_binomial_rng(n1[i],
                    mu1_mid * phi, (1.0 - mu1_mid) * phi);
      y2_rep[i] = beta_binomial_rng(n2[i],
                    mu2_mid * phi, (1.0 - mu2_mid) * phi);
    }
  }
}
