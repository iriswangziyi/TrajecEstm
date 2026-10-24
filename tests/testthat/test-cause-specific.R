# estimate_ensmallen_grad(): the returned estimate maximizes the paper's cause-specific
# objective (obj1), evaluated in pure R by negloglik_cause() (helper-objectives.R).

for (cc in cause_cases) {
  test_that(paste0(cc$label, ": gradient of the objective is near zero at the estimate"), {
    fit <- fit_cause_case(cc)
    expect_length(fit$est, length(fit$init))
    expect_true(all(is.finite(fit$est)))
    # The C++ objective has no I(0 <= S) indicator. On simulated data S = A + V > 0, so the
    # pure-R objective (which has it) and the C++ objective coincide.
    expect_true(all(fit$data$S > 0))

    g_hat <- fd_grad(fit$f, fit$est)
    g_start <- fd_grad(fit$f, fit$init)
    r <- rel_grad(g_hat, g_start)
    expect_lt(r, GRAD_REL_TOL,
              label = sprintf("relative finite-difference gradient at the estimate (%.3g)", r))
  })

  test_that(paste0(cc$label, ": estimate agrees with optim() on the objective"), {
    fit <- fit_cause_case(cc)
    ref <- optim_reference(fit$f, fit$init)
    expect_equal(ref$convergence, 0L)

    d <- max(abs(fit$est - ref$par))
    expect_lt(d, PAR_TOL,
              label = sprintf("max |package estimate - optim()| (%.3g)", d))
    # The package's estimate is at least as good as optim()'s.
    expect_lte(fit$f(fit$est), ref$value + 1e-8 * abs(ref$value))
  })
}
