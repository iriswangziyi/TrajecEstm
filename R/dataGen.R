
#' Generate Simulated Data With Time-Dependent X
#'
#' Creates a synthetic data set for the trajectory estimator simulation.
#' Each subject enters at left-truncation time A, is observed at baseline
#' and follow-up visits, and may experience a type-j failure or censoring.
#' See dataGen_devnotes.R for historical notes and alternative designs.
#'
#' @param T0param  Weibull hazard parameters: (lambda, beta_X1, beta_X2, nu).
#' @param beta0    True regression coefficients: (beta11, beta12, beta21, beta22, beta31, beta32).
#' @param N        Sample size (number of subjects after truncation).
#' @param scenario Integer 1 or 2 (controls r/g/mu0 design).
#' @param tau      Survivor threshold.
#'
#' @return Long-format data.frame with columns: id, T, A, S, Z, Pi, delta, delPi, X1, X2, V, Y.
#'
#' @export
simData2 <- function(T0param = c(lambda = exp(-5)/2,
                                 beta_X1 = 0.5,
                                 beta_X2 = 0.5,
                                 nu = 2),
                     beta0 = c(beta11 = 1,
                               beta12 = -1,
                               beta21 = 1,
                               beta22 = -1,
                               beta31 = 1,
                               beta32 = -1),
                     N = 500,
                     scenario = 2,
                     tau = 20)
{
    ## Underlying population (oversample to account for truncation)
    N0 <- 5 * N

    ## Covariates
    # X1: time-varying binary, X1(s) = I(s >= threshold), threshold ~ Unif(0, 20)
    # X2: time-invariant ~ Unif(0, 1)
    threshold0 <- runif(N0, min = 0, max = 20)
    X2_0 <- runif(N0, 0, 1)

    ## Event time T0 ~ piecewise Weibull with time-varying X1 (Austin 2012, Sec 3.1.2)
    lambda  <- T0param[1]
    beta_X1 <- T0param[2]
    beta_X2 <- T0param[3]
    nu      <- T0param[4]
    u       <- runif(N0, 0, 1)

    T0 <- ifelse(
        -log(u) < lambda * exp(beta_X2 * X2_0) * (threshold0^nu),
        (-log(u) / (lambda * exp(beta_X2 * X2_0)))^(1 / nu),
        ((-log(u)
          - lambda * exp(beta_X2 * X2_0) * (threshold0^nu)
          + lambda * exp(beta_X2 * X2_0 + beta_X1) * (threshold0^nu)
         ) / (lambda * exp(beta_X2 * X2_0 + beta_X1)))^(1 / nu)
    )

    ## Cause type: Pi ~ Binom(0.5) + 1 => {1, 2}
    Pi0 <- rbinom(N0, size = 1, prob = 0.5) + 1

    ## Truncation time: A0 ~ Unif(0, 17.5), ~40% truncated
    A0 <- runif(N0, 0, 17.5)

    pop <- data.frame(A0 = A0, T0 = T0, Pi0 = Pi0, X1 = 0, X2 = X2_0)

    ## Sample: keep subjects with A0 < T0 (left-truncation condition)
    ind <- A0 < T0
    sam.data <- pop[ind == TRUE, ]
    sam.data <- sam.data[1:N, ]

    A         <- sam.data$A0
    T         <- sam.data$T0
    Pi        <- sam.data$Pi0
    X1        <- sam.data$X1
    X2        <- sam.data$X2
    threshold <- threshold0[ind == TRUE][1:N]

    ## Censoring: C ~ Exp(0.04), rounded to nearest even integer (aligns with visit grid)
    C_old <- rexp(N, rate = 0.04)
    C     <- round(C_old) + (round(C_old) %% 2) * (-1)

    ## Observed time and event indicator
    Z     <- ifelse(A + C < T, A + C, T)
    Delta <- ifelse(T <= A + C, 1, 0)

    ## Follow-up visits: V = 0, 2, 4, ..., 20
    V  <- rep(seq(0, 20, 2), N)
    nv <- length(V) / N
    id <- rep(seq_len(N), each = nv)

    ## Expand to long format
    A_l     <- rep(A, each = nv)
    Z_l     <- rep(Z, each = nv)
    X1_l    <- ifelse(A_l + V < rep(threshold, each = nv), 0, 1)
    X2_l    <- rep(X2, each = nv)
    T_l     <- rep(T, each = nv)
    Pi_l    <- rep(Pi, each = nv)
    delta_l <- rep(Delta, each = nv)
    delPi_l <- rep(Delta * Pi, each = nv)
    S       <- A_l + V

    ## Observation error ~ Gamma(shape=10, rate=10) => mean=1, var=0.1
    y_err <- rgamma(N * nv, 10, 10)

    ## Marker trajectory: Y(s) = mu_j(s,t) * exp(beta'X) * gamma_s
    g_sig <- 4

    if (scenario == 1) {
        # Scenario 1 (both DOWN)
        rs1  <- compute_r_vec(S, T_l, c(-1, -1), sce = 2.2, tau = tau, center = 0.5)
        gs1  <- 0.5 + 0.1 * T_l
        mus1 <- rs1 * gs1

        rs2  <- compute_r_vec(S, T_l, c(0.5, 1.0), sce = 1.2, tau = tau, center = 0.5)
        gs2  <- 0.5
        mus2 <- rs2 * gs2

        mus0 <- 5 * exp(-log(10) / 30 * S)
    } else {
        # Scenario 2 (both UP)
        rs1  <- compute_r_vec(S, T_l, 2, sce = 2.1, tau = tau, center = 0)
        gs1  <- g_sig
        mus1 <- rs1 * gs1

        rs2  <- compute_r_vec(S, T_l, c(0.5, 0.5), sce = 1.1, tau = tau, center = 0.5)
        gs2  <- 4 * exp(-0.05 * T_l)
        mus2 <- rs2 * gs2

        mus0 <- 0.5 * exp(log(6) / 30 * S)
    }

    ## Beta coefficients
    beta11 <- beta0[1]; beta12 <- beta0[2]
    beta21 <- beta0[3]; beta22 <- beta0[4]
    beta31 <- beta0[5]; beta32 <- beta0[6]

    ## Generate Y for each group
    Y <- numeric(length(S))

    idx1 <- (Pi_l == 1) & (T_l < tau)
    idx2 <- (Pi_l == 2) & (T_l < tau)
    idx3 <- (T_l >= tau)

    Y[idx1] <- y_err[idx1] * mus1[idx1] * exp(X1_l[idx1] * beta11 + X2_l[idx1] * beta12)
    Y[idx2] <- y_err[idx2] * mus2[idx2] * exp(X1_l[idx2] * beta21 + X2_l[idx2] * beta22)
    Y[idx3] <- y_err[idx3] * mus0[idx3] * exp(X1_l[idx3] * beta31 + X2_l[idx3] * beta32)

    ## Assemble long-format data, keep only observed visits (S <= Z)
    df2 <- data.frame(
        id = id, T = T_l, A = A_l, S = S, Z = Z_l,
        Pi = Pi_l, delta = delta_l, delPi = delPi_l,
        X1 = X1_l, X2 = X2_l, V = V, Y = Y
    )

    df2[S <= Z_l, ]
}
