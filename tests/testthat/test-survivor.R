# estimate_survivor_grad(): the returned estimate maximizes the paper's survivor objective
# (surv_obj), evaluated in pure R by negloglik_surv() (helper-objectives.R).
#
# Both checks fail if the analytic gradient in the C++ objective is wrong, for example if the
# kernel weights are normalized by the wrong denominator: L-BFGS then stops at a point that is
# not a stationary point of the objective.

for (scenario in 1:2) {
  label <- c("Scenario I", "Scenario II")[scenario]

  test_that(paste0("Survivor, ", label, ": gradient of the objective is near zero at the estimate"), {
    fit <- fit_surv_case(scenario)
    expect_length(fit$est, 2L)
    expect_true(all(is.finite(fit$est)))

    g_hat <- fd_grad(fit$f, fit$est)
    g_start <- fd_grad(fit$f, fit$init)
    r <- rel_grad(g_hat, g_start)
    expect_lt(r, GRAD_REL_TOL,
              label = sprintf("relative finite-difference gradient at the estimate (%.3g)", r))
  })

  test_that(paste0("Survivor, ", label, ": estimate agrees with optim() on the objective"), {
    fit <- fit_surv_case(scenario)
    ref <- optim_reference(fit$f, fit$init)
    expect_equal(ref$convergence, 0L)

    d <- max(abs(fit$est - ref$par))
    expect_lt(d, PAR_TOL,
              label = sprintf("max |package estimate - optim()| (%.3g)", d))
    expect_lte(fit$f(fit$est), ref$value + 1e-8 * abs(ref$value))
  })
}
