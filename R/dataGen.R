
# g()


#' Generate Simulated Data With Time-Dependent X
#'
#' Creates a synthetic data set suitable for testing the trajectory
#' estimator.  Each subject enters the study at left-truncation time
#' \eqn{A_0}, is observed at baseline and at \code{m} follow-up visits,
#' and may experience a type-\code{j} failure or censoring at time
#' \eqn{Z}.  The marker trajectory is generated under Model (I) with
#' user-specified parameters.
#'
#' @param n         Integer. Number of subjects to simulate.
#' @param p         Integer. Dimension of the covariate vector \code{X}.
#' @param m         Integer. Number of follow-up visits after baseline.
#' @param beta      Numeric vector of length \code{p}. True regression coefficients.
#' @param theta     Numeric. True value of \eqn{\theta}.
#' @param seed      Integer (optional). RNG seed for reproducibility.
#'
#' @return A list with elements
#'   \describe{
#'     \item{\code{X}}{p × n design matrix (columns = subjects).}
#'     \item{\code{A}}{Vector of left-truncation times (length n).}
#'     \item{\code{V}}{List of visit-offset vectors, one per subject.}
#'     \item{\code{Z}}{Vector of failure/censoring times (length n).}
#'     \item{\code{Y}}{Vector of marker values at each recorded visit.}
#'     \item{\code{Pi}}{Failure-type indicator (length n).}
#'     \item{\code{Delta}}{Censoring indicator (length n).}
#'   }
#'
#' @examples
#' demo <- simData2(n = 100, p = 3, m = 2,
#'                  beta = c(0.3, -0.5, 1.0),
#'                  theta = 0.8, seed = 1)
#' str(demo)
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

    ##tau for surv model
    #tau = 20

    ## underlying population
    N0 <- 5*N
    ## Generate X1 and X2
    # X1_0 is time-varying covariate ~binary, jump from 0 to 1 at threshold0
    # X2_0 fixed time-invariant covariates ~follow unif(0, 1)
    threshold0 <- runif(N0, min = 0, max = 5) #threshold0=t0 in Austin's
    # X1_0 <- ifelse(0 >= threshold0, 1, 0)
    X2_0 <- runif(N0, 0, 1)

    ## Generate T0
    # Ref: Austin 2021, 3.1.2 Weibull dist. of event time
    # T0param = (lambda, beta_X1, beta_X2, nu)
    # lambda: baseline event rate (hazard), higher lambda, events happen sooner
    # beta_X1: regression coeff. asso. w/ X1_0
    # beta_X2: regression coeff. asso. w/ X2_0
    # nu: shape param (lambda is scale param)

    #T0param_old <- c(lambda = 0.1, beta_X1 = 0.5, beta_X2 = 0.3) #e.g

    #T0param <- c(lambda = exp(-6), beta_X1 = 0.5, beta_X2 = 0.5, nu = 2) #e.g

    lambda <- T0param[1]
    beta_X1 <- T0param[2]
    beta_X2 <- T0param[3]
    nu <- T0param[4]
    u <- runif(N0, 0, 1)

    #Weibull
    T0 <- ifelse(
        -log(u) < lambda * exp(beta_X2 * X2_0) * (threshold0^nu),

        # before threshold: T = [ -log(u) / (lambda exp(eta0)) ]^(1/nu)
        (-log(u) / (lambda * exp(beta_X2 * X2_0)))^(1/nu),

        # after threshold:
        # T = [ ( -log(u) - lambda exp(eta0) t0^nu + lambda exp(eta1) t0^nu )
        #       / (lambda exp(eta1)) ]^(1/nu)
        (
            (-log(u)
             - lambda * exp(beta_X2 * X2_0) * (threshold0^nu)
             + lambda * exp(beta_X2 * X2_0 + beta_X1) * (threshold0^nu)
            ) /
                (lambda * exp(beta_X2 * X2_0 + beta_X1))
        )^(1/nu)
    )
    #hist(T0, main="T0 Weibull, N=500, N0 = 5N")

    # T0_old is Exponential Event times
    # T0_old <- ifelse(
    #     -log(u) < lambda * exp(beta_X2 * X2_0) * threshold0,
    #     (-log(u)) / (lambda * exp(beta_X2 * X2_0)),
    #     (-log(u)
    #      - lambda * exp(beta_X2 * X2_0) * threshold0
    #      + lambda * exp(beta_X2 * X2_0 + beta_X1) * threshold0) /
    #         (lambda * exp(beta_X2 * X2_0 + beta_X1))
    # )

    ## Generate Pi0 - cause
    Pi0 <- rbinom(N0, size = 1, prob = 0.5) + 1  # Generates 1 or 2

    ### if cause depends on X1
    # i.e. If X1_0 = 1, then 70% chance of Cause 1, 30% of Cause 2.
     #p_cause1 <- ifelse(X1_0 == 1, 0.7, 0.3)
     #Pi0 <- rbinom(N0, size = 1, prob = p_cause1) + 1

    ### if cause depends on X2
    # i.e. If higher X2 makes Cause 1 more likely
    # p_cause1 <- plogis(2 * (X2_0 - 0.5))  # Logistic function for smooth transition
    # Pi0 <- rbinom(N0, size = 1, prob = p_cause1) + 1

    ## Generate truncation time: A0
    # 7.18 used to be 3.3, truncation_prob = 0.2
    #with exp T0, now changed to 10, truncation_prob = 0.4899236
    #with Weibull T0, now changed to 17.5, truncation_prob = 0.4086
    A0 <- runif(N0, 0, 17.5)

    # WANT: truncation_prob = P(T0<A0), sum(T0<=A0)/N0~0.25
    #truncation_prob = sum(T0 <= A0) / N0 #0.4086
    #truncation_prob

    pop <- data.frame(A0 = A0, T0 = T0, Pi0 = Pi0, X1 = 0, X2 = X2_0)

    ## sample data (n=N): A0<T0
    ind <- A0 < T0
    sam.data <- pop[ind == TRUE,]
    sam.data <- sam.data[1:N,]

    ### This is my sample data
    A <- sam.data$A0
    T <- sam.data$T0
    Pi <- sam.data$Pi0
    X1 <- sam.data$X1
    X2 <- sam.data$X2
    threshold <- threshold0[ind == TRUE][1:N]

    ## residual censoring time after enrollment: C
    C_old <- rexp(N, rate = 0.04)

    #9.30
    #round to closest small even int
    C <- round(C_old) + (round(C_old) %%2) * (-1)

    # WANT: censoring_rate = P(A+C<T) ~0.2
    #censoring_rate <- sum(A + C < T) / N #0.219027
    #censoring_rate

    ## Generating Z: observed censored event time
    ### Z = min(T, A+C)
    Z <- ifelse(A + C < T, A + C, T)

    ## Generating Delta: event indicator
    ### Delta = I(T <= A+C)
    Delta <- ifelse(T <= A + C, 1, 0)

    ## Generating V: follow-up visits
    # V0 = 0, V1 = 2, V2 = 4, ...
    # keep V <= Z-A
    V <- rep(seq(0, 20, 2), N)
    nv <- length(V)/N
    id <- rep(seq(1:N), each = nv)

    A_l <- rep(A, each = nv)
    Z_l <- rep(Z, each = nv)


    ## X1
    X1_l <- ifelse(V < rep(threshold, each = nv), 0, 1)
    X2_l <- rep(X2, each = nv)
    T_l <- rep(T, each = nv)
    Pi_l <- rep(Pi, each = nv)
    delta_l <- rep(Delta, each = nv)
    delPi_l <- rep(Delta*Pi, each = nv)

    ## Generate S: measurement time (A<=S<=Z)
    S <- A_l + V

    ## Generating Y(s)
    # Observation error - Gamma(mean=1, var=0.1)
    # Shape=10, Rate=10 ensures mean=1, var=0.1
    y_err <- rgamma(N*nv, 10, 10)

    ## Y(s) = mu_j(s,t;θ)*exp(betaTX)*y_err
    ## Subject get type j event at time t, obsed at time s.
    # baseline trajectory: muj0(s,t;θ)=gj0(t)*rj(s,t;θ)

    ## 1.gamma: rj(s,t;θ) = fj (s;0)/Fj (t;0), gj0(t) = 1
    ## true theta: cause1:2, cause2:5
    ## 2. sigmoid function: rj(s,t;θ) = 1/(1 + e^{θ(s-t/2)}),gj0(t)=1 or 1/(1+t)
    ## true theta: cause1:1, cause2:2

    # Compute marker trajectory mu_j(s,t;θ) = g_j(t) * r_j(s,t;θ)
    # mu0: survivor baseline (chosen independently by visual inspection)
    mu0_sce1 <- 4; mu0_sce2 <- 0.5

    # g(t) scaling constant for Sce 2: (1+e)
    g_sig <- 1 + exp(1)  # ~ 3.72

    if (scenario == 1) {
        # Scenario 1 (both DOWN):
        # j=1: s+s² (sce 2.2), centered c=0.5, DOWN
        rs1 <- compute_r_vec(s = S, t = T_l, theta = c(-1, -1),
                             sce = 2.2, tau = tau, center = 0.5)
        gs1 <- 0.5 + 0.1 * T_l
        mus1 <- rs1 * gs1

        # j=2: s+(t-s) (sce 1.2), theta=(-1,1), no centering, DOWN
        rs2 <- compute_r_vec(s = S, t = T_l, theta = c(-1, 1),
                             sce = 1.2, tau = tau)
        gs2 <- 1
        mus2 <- rs2 * gs2

        # long-term surv
        mus0 <- rep(mu0_sce1, length(S))

    }else{
        # Scenario 2 (both UP):
        # j=1: Sigmoid (sce 2.1), UP
        rs1 <- compute_r_vec(s = S, t = T_l, theta = 2,
                             sce = 2.1, tau = tau)
        gs1 <- g_sig
        mus1 <- rs1 * gs1

        # j=2: s+s×t (sce 1.1), no centering, UP
        rs2 <- compute_r_vec(s = S, t = T_l, theta = c(1, -0.5),
                             sce = 1.1, tau = tau)
        gs2 <- g_sig / (1 + exp(T_l / tau))
        mus2 <- rs2 * gs2

        # long-term surv
        mus0 <- rep(mu0_sce2, length(S))
    }

    ## Extracting true values for beta parameters
    # Cause 1
    beta11 <- beta0[1]; beta12 <- beta0[2]
    # Cause 2
    beta21 <- beta0[3]; beta22 <- beta0[4]
    # Cause 3 - surv
    beta31 <- beta0[5]; beta32 <- beta0[6]


    # Preallocate Y for all rows
    Y <- numeric(length(S))  # same length as S, T_l, Pi, y_err, mus*

    # Define indices for each type
    #type1: df_l$Pi == 1 & df_l$T_l < tau
    #type2: df_l$Pi == 2 & df_l$T_l < tau
    #type3: df_l$T_l > tau, we use=> mus0

    idx1 <- (Pi_l == 1) & (T_l < tau)       # type 1: cause 1 before tau
    idx2 <- (Pi_l == 2) & (T_l < tau)       # type 2: cause 2 before tau
    idx3 <- (T_l >= tau)                    # type 3: long-term survivors

    # Compute Y(s) for each cause
    # Y <- y_err*muj*exp(X1*beta1j + X2*beta2j)
    Y[idx1] <- y_err[idx1] * mus1[idx1] *
        exp(X1_l[idx1] * beta11 + X2_l[idx1] * beta12)

    Y[idx2] <- y_err[idx2] * mus2[idx2] *
        exp(X1_l[idx2] * beta21 + X2_l[idx2] * beta22)

    Y[idx3] <- y_err[idx3] * mus0[idx3] *
        exp(X1_l[idx3] * beta31 + X2_l[idx3] * beta32)  # your chosen betas for surv

    df2 <- data.frame(
        id = id, T = T_l,
        A = A_l, S = S, Z = Z_l,
        Pi = Pi_l, delta = delta_l, delPi = delPi_l,
        X1 = X1_l, X2 = X2_l, V = V,
        Y = Y
    )

    ind2 <- S <= Z_l
    df_l <- df2[ind2, ]

    df_l

    ## sample (with follow-up visit)
    # id: 1:N
    # T: failure time
    # A: truncation time (baseline time)
    # S: measurement time (A<=S<=Z) #[check?:S = a+v]
    # Z: observed survival time
    # Pi: cause of failure: 1 or 2
    # delta: censoring indicator: 1: event, 0: cens
    # delPi: delta*Pi: uncensoring cause of failure
    # X1, X2: two covariates
    # V: measurement point
    # Y: marker at time S in one of the two failure causes


    ## sample (baseline only)
    # df <- df_l[df_l$V == 0,]
    #
    # out <- list()
    # out$df <- df
    # out$df_l <- df_l
    #
    # return(out)
}
