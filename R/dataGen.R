
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

simData2 <- function(T0param = c(lambda = 0.1,
                                 beta_X1 = 0.5,
                                 beta_X2 = 0.3),
                     beta0 = c(beta11 = .5,
                               beta12 = -0.5,
                               beta21 = .5,
                               beta22 = -0.5),
                     N = 100,
                     scenario = 2)
{

    ## underlying population
    N0 <- 2*N
    ## Generate X1 and X2
    # X1_0 is time-varying covariate ~binary, jump from 0 to 1 at threshold0
    # X2_0 fixed time-invariant covariates ~follow unif(0, 1)
    threshold0 <- runif(N0, min = 0, max = 5) #threshold0=t0 in Austin's
    # X1_0 <- ifelse(0 >= threshold0, 1, 0)
    X2_0 <- runif(N0, 0, 1)

    ## Generate T0
    # Ref: Austin 2021, 3.1.1 Exponential dist. of event time
    # T0param = (lambda, beta_X1, beta_X2)
    # lambda: baseline event rate (hazard), higher lambda, events happen sooner
    # beta_X1: regression coeff. asso. w/ X1_0
    # beta_X2: regression coeff. asso. w/ X2_0

    #T0param <- c(0.1, 0.5, 0.3) #e.g1
    #T0param <- c(0.5, 0.1, 0.5) #e.g2

    lambda <- T0param[1]
    beta_X1 <- T0param[2]
    beta_X2 <- T0param[3]
    #u ~ unif(0,1)
    u <- runif(N0, 0, 1)
    T0 <- ifelse(
        -log(u) < lambda * exp(beta_X2 * X2_0) * threshold0,
        (-log(u)) / (lambda * exp(beta_X2 * X2_0)),
        (-log(u)
         - lambda * exp(beta_X2 * X2_0) * threshold0
         + lambda * exp(beta_X2 * X2_0 + beta_X1) * threshold0) /
            (lambda * exp(beta_X2 * X2_0 + beta_X1))
    )
    #hist(T0)

    ## Generate Pi0 - cause
    Pi0 <- rbinom(N0, size = 1, prob = 0.5) + 1  # Generates 1 or 2

    ### if cause depends on X1
    # i.e. If X1_0 = 1, then 70% chance of Cause 1, 30% of Cause 2.
    # p_cause1 <- ifelse(X1_0 == 1, 0.7, 0.3)
    # Pi0 <- rbinom(N0, size = 1, prob = p_cause1) + 1

    ### if cause depends on X2
    # i.e. If higher X2 makes Cause 1 more likely
    # p_cause1 <- plogis(2 * (X2_0 - 0.5))  # Logistic function for smooth transition
    # Pi0 <- rbinom(N0, size = 1, prob = p_cause1) + 1

    ## Generate truncation time: A0
    A0 <- runif(N0, 0, 3.3)

    # WANT: truncation_prob = P(T0<A0) ~0.25, sum(T0<=A0)/N0~0.25
    truncation_prob = sum(T0 <= A0) / N0 # 0.2
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

    ## residual censoring time after enrollment: C
    C <- rexp(N, rate = 0.04)

    # WANT: censoring_rate = P(A+C<T) ~0.25
    censoring_rate <- sum(A + C < T) / N #0.22
    #censoring_rate

    ## Generating Z: observed censored event time
    ### Z = min(T, A+C)
    Z <- ifelse(A + C < T, A + C, T)

    ## Generating Delta: event indicator
    ### Delta = I(T <= A+C)
    Delta <- ifelse(T <= A + C, 1, 0)

    ## Generating V: follow-up visits
    # V0 = 0, V1 = 1, V2 = 2, ...,V5 = 5
    # keep V <= Z-A
    V <- rep(seq(0, 5, 1), N)
    nv <- 6
    id <- rep(seq(1:N), each = nv)

    A_l <- rep(A, each = nv)
    Z_l <- rep(Z, each = nv)
    ## X1
    X1_l <- ifelse(V < threshold0, 0, 1)
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

    # Compute marker trajectory μ_j(s,t;θ) = g_j(t) * r_j(s,t;θ)
    if (scenario == 1) {
        # Scenario 1.1: Gamma PDF
        rs1 <- compute_r_vec(s = S, t = T_l, theta = log(0.5), sce = 1.1)
        gs1 <- 1
        mus1 <- rs1 * gs1

        # Scenario 1.2: Poly
        rs2 <- compute_r_vec(s = S, t = T_l, theta = c(0.2, 0.1), sce = 1.2)
        gs2 <- 1 + 0.1 * T_l
        mus2 <- rs2 * gs2

    }else{
        ## Scenario 2.1:Sigmoid
        rs1 <- compute_r_vec(s = S, t = T_l, theta = 1.2, sce = 2.1)
        gs1 <- log(T_l + 2)
        mus1 <- rs1 * gs1

        # Scenario 2.2: Weibull
        rs2 <- compute_r_vec(s = S, t = T_l, theta = log(c(0.8, 2)), sce = 2.2)
        gs2 <- sqrt(T_l + 1)
        mus2 <- rs2 * gs2
    }

    ## Extracting true values for beta parameters
    # Cause 1
    beta11 <- beta0[1]
    beta12 <- beta0[2]
    # Cause 2
    beta21 <- beta0[3]
    beta22 <- beta0[4]

    # Compute Y(s) for each cause
    # Y <- y_err*muj*exp(X1*beta1j + X2*beta2j)
    Y1 <- y_err*mus1*exp(X1_l*beta11 + X2_l*beta12) # Cause 1
    Y2 <- y_err*mus2*exp(X1_l*beta21 + X2_l*beta22) # Cause 2

    ## Filtering: Keep only valid observations where S ≤ Z_l
    ind2 <- S <= Z_l

    df2 <- data.frame(id = id, T = T_l,
                      A = A_l, S = S, Z = Z_l,
                      Pi = Pi_l, delta = delta_l, delPi = delPi_l,
                      Y1 = Y1, Y2 = Y2, X1 = X1_l, X2 = X2_l, V = V)

    # only keep S=A+V<=Z (ind2=TRUE)
    df_l <- df2[ind2 == TRUE,]
    # Y=Y1 if cause1, Y=Y2 if cause2
    df_l$Y <- ifelse(df_l$Pi == 1, df_l$Y1, df_l$Y2)
    df_l$Y1 <- NULL
    df_l$Y2 <- NULL

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
