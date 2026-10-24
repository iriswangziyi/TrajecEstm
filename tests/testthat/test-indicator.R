# The cause-specific estimator and mu-hat exclude visits before the reference age (S < 0), as the
# indicator I(0 <= A + V) in the paper. Fitting data with some S < 0 rows must give exactly the same
# result as fitting the same data with those rows removed. The survivor estimator is unchanged.

indicator_data <- function() {
  df <- sim_long(scenario = 1, n = TEST_N, seed = TEST_SEEDS[1])
  # Move a few type-1 visits before the reference age.
  idx <- which(df$delPi == 1)[c(1, 4, 9, 15)]
  df$S[idx] <- -df$S[idx] - 1
  df
}

test_that("estimate_ensmallen_grad() ignores visits with S < 0", {
  df <- indicator_data()
  X <- rbind(df$X1, df$X2)
  h <- cause_bandwidth(TEST_N)
  args <- list(j = 1, h1 = h, tau0 = ceiling(h), tau1 = floor(TEST_TAU - h), tau_norm = TEST_TAU,
               sce = 2.2, init = rep(0, 4), center = 0.5)
  fit_all <- do.call(estimate_ensmallen_grad, c(list(df_l = df, X_l = X), args))
  keep <- df$S >= 0
  fit_pos <- do.call(estimate_ensmallen_grad, c(list(df_l = df[keep, ], X_l = X[, keep]), args))
  expect_true(any(!keep))
  expect_equal(as.numeric(fit_all), as.numeric(fit_pos), tolerance = 1e-12)
})

test_that("mu_hat() and mu_hat_fast() ignore visits with S < 0", {
  df <- indicator_data()
  X <- rbind(df$X1, df$X2)
  par <- c(1, -1, -1, -1)
  keep <- df$S >= 0
  for (f in list(mu_hat, mu_hat_fast)) {
    a <- f(1, 10, 5, 2, par, X, df$Y, df$delPi, df$S, df$Z, 2.2, TEST_TAU, 0.5)
    b <- f(1, 10, 5, 2, par, X[, keep], df$Y[keep], df$delPi[keep], df$S[keep], df$Z[keep],
           2.2, TEST_TAU, 0.5)
    expect_equal(a, b, tolerance = 1e-12)
  }
})
