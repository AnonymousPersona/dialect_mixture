// Null / intercept-only beta-binomial model for alpha / eta variants

data {
  // ── Outcome counts per inscription ──────────────────────────────────
  int<lower=1> N;

  array[N] int<lower=0> y1;   // alpha tokens in outer context
  array[N] int<lower=0> n1;   // total tokens in outer context
  array[N] int<lower=0> y2;   // alpha tokens in inner context
  array[N] int<lower=0> n2;   // total tokens in inner context
}

parameters {
  real           alpha;     // global intercept (log-odds of alpha, outer)
  real           gamma_ctx; // global inner context shift
  real<lower=0>  phi;       // beta-binomial concentration
}

model {
  // ── Priors (identical to STGP models) ───────────────────────────────
  alpha     ~ normal(0, 1);
  gamma_ctx ~ normal(0, 2);
  phi       ~ lognormal(log(5), 0.6);

  // ── Likelihood ───────────────────────────────────────────────────────
  // mu1 and mu2 are constant across inscriptions (no inscription-level
  // covariates), so the beta-binomial shape parameters are computed once
  // outside the loop.
  //
  // Terms where n = 0 and y = 0 contribute exactly 0:
  //   lchoose(0, 0) = 0
  //   lbeta(a + 0, b + 0) - lbeta(a, b) = 0
  // so no explicit guard is needed.
  {
    real mu1 = inv_logit(alpha);
    real a1  = mu1 * phi;
    real b1  = (1.0 - mu1) * phi;

    real mu2 = inv_logit(alpha + gamma_ctx);
    real a2  = mu2 * phi;
    real b2  = (1.0 - mu2) * phi;

    for (i in 1:N) {
      target +=
          lchoose(n1[i], y1[i])
        + lbeta(a1 + y1[i], b1 + n1[i] - y1[i]) - lbeta(a1, b1)
        + lchoose(n2[i], y2[i])
        + lbeta(a2 + y2[i], b2 + n2[i] - y2[i]) - lbeta(a2, b2);
    }
  }
}

generated quantities {
  vector[N]                    log_lik;
  array[N] int                 y1_rep;
  array[N] int                 y2_rep;
  vector<lower=0, upper=1>[N] theta_bar_out;   // P(alpha) in outer
  vector<lower=0, upper=1>[N] theta_bar_in;    // P(alpha) in inner
  vector<lower=0, upper=1>[N] theta_bar;       // n-weighted average

  // mu1 and mu2 are constant across inscriptions; compute once.
  real mu1_gq = inv_logit(alpha);
  real mu2_gq = inv_logit(alpha + gamma_ctx);
  real a1_gq  = mu1_gq * phi;
  real b1_gq  = (1.0 - mu1_gq) * phi;
  real a2_gq  = mu2_gq * phi;
  real b2_gq  = (1.0 - mu2_gq) * phi;

  for (i in 1:N) {
    log_lik[i] =
        lchoose(n1[i], y1[i])
      + lbeta(a1_gq + y1[i], b1_gq + n1[i] - y1[i]) - lbeta(a1_gq, b1_gq)
      + lchoose(n2[i], y2[i])
      + lbeta(a2_gq + y2[i], b2_gq + n2[i] - y2[i]) - lbeta(a2_gq, b2_gq);

    // theta_bar_out and theta_bar_in are identical for every inscription
    // in the null model (no inscription-level covariates).
    theta_bar_out[i] = mu1_gq;
    theta_bar_in[i]  = mu2_gq;

    {
      real denom = n1[i] + n2[i];
      theta_bar[i] = (denom > 0)
                   ? (n1[i] * theta_bar_out[i] + n2[i] * theta_bar_in[i]) / denom
                   : 0.5;
    }

    y1_rep[i] = beta_binomial_rng(n1[i], a1_gq, b1_gq);
    y2_rep[i] = beta_binomial_rng(n2[i], a2_gq, b2_gq);
  }
}
