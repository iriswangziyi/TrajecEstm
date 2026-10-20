# compute_r_dr(): r matches the paper's shape functions (log_r_shape() in helper-objectives.R)
# and the Jacobian dr matches central finite differences of r in theta.

r_test_points <- function() {
  with_test_seed(303, {
    s <- stats::runif(40, 0, 30)
    list(s = s, t = s + stats::runif(40, 0, 20))
  })
}

for (code in c(2.1, 2.2, 1.1, 1.2)) {
  test_that(sprintf("compute_r_dr, code %s: r and the Jacobian dr/dtheta", code), {
    pts <- r_test_points()
    q <- n_theta(code)
    thetas <- with_test_seed(404, replicate(3, stats::runif(q, -2, 2), simplify = FALSE))

    for (tau in c(18, 28)) {
      for (center in c(0, 0.5)) {
        for (theta in thetas) {
          out <- compute_r_dr(pts$s, pts$t, theta, sce = code, tau = tau, center = center)
          r <- as.numeric(out$r)
          expect_equal(dim(out$dr), c(length(pts$s), q))

          r_paper <- exp(log_r_shape(pts$s, pts$t, theta, code, tau, center))
          expect_equal(r, r_paper, tolerance = 1e-12)
          r_vec <- compute_r_vec(pts$s, pts$t, theta, sce = code, tau = tau, center = center)
          expect_equal(r, as.numeric(r_vec), tolerance = 1e-14)

          jac_fd <- fd_jacobian(function(th) {
            as.numeric(compute_r_dr(pts$s, pts$t, th, sce = code, tau = tau, center = center)$r)
          }, theta)
          err <- max(abs(out$dr - jac_fd) / pmax(1, abs(jac_fd)))
          expect_lt(err, JAC_REL_TOL,
                    label = sprintf("relative Jacobian error (%.3g; tau %g, center %g)",
                                    err, tau, center))
        }
      }
    }
  })
}

test_that("compute_r_dr rejects unsupported shape codes", {
  expect_error(compute_r_dr(1, 2, c(1, 1), sce = 3, tau = 18), "Unsupported sce")
})
