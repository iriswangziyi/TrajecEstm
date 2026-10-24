# Small simulated data sets for the estimator tests. The settings follow the paper's simulation
# design (Section 5): tau = 18, beta = (1, -1) in every group, Epanechnikov kernel, bandwidth
# h = sigma * n_eff^(-1/3) with sigma = 4.142 and n_eff = round(0.298 n) for the cause-specific
# models, sigma = 4.986 and n_eff = round(0.164 * 6.17 n) for the survivor model, and the
# boundary window [ceiling(h), floor(tau - h)].

# Run expr with a fixed seed and restore the caller's random number stream afterwards.
with_test_seed <- function(seed, expr) {
  had_seed <- exists(".Random.seed", envir = globalenv(), inherits = FALSE)
  if (had_seed) old_seed <- get(".Random.seed", envir = globalenv(), inherits = FALSE)
  on.exit({
    if (had_seed) {
      assign(".Random.seed", old_seed, envir = globalenv())
    } else if (exists(".Random.seed", envir = globalenv(), inherits = FALSE)) {
      rm(".Random.seed", envir = globalenv())
    }
  })
  set.seed(seed)
  expr
}

TEST_TAU <- 18
TEST_N <- 250
TEST_SEEDS <- c(101, 202)  # Scenario I, Scenario II

sim_long <- function(scenario, n, seed) {
  with_test_seed(seed, simData2(
    T0param = c(lambda = exp(-5) / 2, beta_X1 = 0.5, beta_X2 = 0.5, nu = 2),
    beta0 = c(beta11 = 1, beta12 = -1, beta21 = 1, beta22 = -1, beta31 = 1, beta32 = -1),
    N = n, scenario = scenario, tau = TEST_TAU
  ))
}

# Rows of subjects with an observed type-j failure, in the form used by negloglik_cause().
cause_setup <- function(df, X, j, code, center, h, tau0, tau1, tau_norm) {
  keep <- df$delPi == j
  d <- list(X = X[, keep, drop = FALSE], Y = df$Y[keep], S = df$S[keep], Z = df$Z[keep],
            h = h, tau0 = tau0, tau1 = tau1, tau_norm = tau_norm, code = code, center = center)
  d$K <- epan_kernel(outer(d$Z, d$Z, "-"), h)
  d
}

# Rows of long-term survivors (Z >= tau), in the form used by negloglik_surv().
surv_setup <- function(df, X, h, tau, tau0, tau1) {
  keep <- df$Z >= tau
  d <- list(X = X[, keep, drop = FALSE], Y = df$Y[keep], S = df$S[keep],
            h = h, tau0 = tau0, tau1 = tau1)
  d$K <- epan_kernel(outer(d$S, d$S, "-"), h)
  d
}

cause_bandwidth <- function(n) 4.142 * round(0.298 * n)^(-1 / 3)
surv_bandwidth <- function(n) 4.986 * round(0.164 * n * 6.17)^(-1 / 3)

# The cause-specific fits under test: the paper's shape code for each cause in each scenario,
# plus code 1.1 in the configuration of the BIOCARD analysis (three covariates, no boundary
# trimming: tau0 = 0, tau1 = tau_norm), fitted here to simulated data.
cause_cases <- list(
  list(label = "Scenario I, cause 1, code 2.2", scenario = 1, j = 1, code = 2.2, center = 0.5),
  list(label = "Scenario I, cause 2, code 1.2", scenario = 1, j = 2, code = 1.2, center = 0.5),
  list(label = "Scenario II, cause 1, code 2.1", scenario = 2, j = 1, code = 2.1, center = 0),
  list(label = "Scenario II, cause 2, code 1.1", scenario = 2, j = 2, code = 1.1, center = 0.5),
  list(label = "BIOCARD configuration, code 1.1", scenario = 1, j = 1, code = 1.1, center = 0.5,
       biocard = TRUE)
)

# Builds the data, runs the package estimator and returns what a test needs.
fit_cause_case <- function(cc, n = TEST_N) {
  seed <- TEST_SEEDS[[cc$scenario]]
  df <- sim_long(cc$scenario, n, seed)
  X <- rbind(df$X1, df$X2)
  h <- cause_bandwidth(n)
  if (isTRUE(cc$biocard)) {
    # A third, subject-level binary covariate with no effect on the marker.
    ids <- unique(df$id)
    carrier <- with_test_seed(seed + 1, stats::rbinom(length(ids), 1, 0.3))
    X <- rbind(X, carrier[match(df$id, ids)])
    tau0 <- 0
    tau1 <- TEST_TAU
  } else {
    tau0 <- ceiling(h)
    tau1 <- floor(TEST_TAU - h)
  }
  m <- nrow(X) + n_theta(cc$code)
  est <- as.numeric(estimate_ensmallen_grad(
    j = cc$j, df_l = df, X_l = X, h1 = h, tau0 = tau0, tau1 = tau1, tau_norm = TEST_TAU,
    sce = cc$code, init = rep(0, m), center = cc$center
  ))
  d <- cause_setup(df, X, cc$j, cc$code, cc$center, h, tau0, tau1, TEST_TAU)
  list(case = cc, est = est, init = rep(0, m), data = d,
       f = function(par) negloglik_cause(par, d))
}

fit_surv_case <- function(scenario, n = TEST_N) {
  df <- sim_long(scenario, n, TEST_SEEDS[[scenario]])
  X <- rbind(df$X1, df$X2)
  h <- surv_bandwidth(n)
  tau0 <- ceiling(h)
  tau1 <- floor(TEST_TAU - h)
  est <- as.numeric(estimate_survivor_grad(
    df_s = df, X_s = X, h1 = h, tau0 = tau0, tau1 = tau1, tau = TEST_TAU, init = rep(0, 2)
  ))
  d <- surv_setup(df, X, h, TEST_TAU, tau0, tau1)
  list(scenario = scenario, est = est, init = rep(0, 2), data = d,
       f = function(beta) negloglik_surv(beta, d))
}
