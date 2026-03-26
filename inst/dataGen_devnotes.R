# ============================================================
# dataGen_devnotes.R
# Historical notes for simData2() in dataGen.R
# NOT part of the package — kept for reference only.
# ============================================================

# --- T0 distribution evolution ---
# Original: Exponential event times (T0_old)
#   T0_old <- ifelse(-log(u) < lambda*exp(beta_X2*X2_0)*threshold0, ...)
# Changed to Weibull on 1/21/26 (Austin 2012, 3.1.2)
#   T0param_old <- c(lambda = 0.1, beta_X1 = 0.5, beta_X2 = 0.3)
#   T0param <- c(lambda = exp(-6), ...)  # earlier try

# --- Cause (Pi) alternatives explored ---
# Default: Pi ~ Binom(0.5)+1 (independent of covariates)
# Tried: cause depends on X1
#   p_cause1 <- ifelse(X1_0 == 1, 0.7, 0.3)
#   Pi0 <- rbinom(N0, 1, p_cause1) + 1
# Tried: cause depends on X2
#   p_cause1 <- plogis(2 * (X2_0 - 0.5))
#   Pi0 <- rbinom(N0, 1, p_cause1) + 1

# --- Truncation A0 evolution ---
# A0~Unif(0, 3.3): trunc_prob ~0.2
# A0~Unif(0, 10): trunc_prob ~0.49 (with Exp T0)
# A0~Unif(0, 17.5): trunc_prob ~0.41 (with Weibull T0) <- current

# --- X1 threshold evolution ---
# 3/20/26: threshold changed from Unif(0,5) to Unif(0,20)
# 3/20/26: X1 switching fixed from V scale to absolute S=A+V scale
# Old: X1_l <- ifelse(V < threshold, 0, 1)
# New: X1_l <- ifelse(A_l+V < threshold, 0, 1)

# --- Censoring ---
# C ~ Exp(0.04), then rounded to nearest even integer (9/30 note)
# C <- round(C_old) + (round(C_old) %% 2) * (-1)
# Rounding ensures censoring times align with visit grid V=0,2,4,...
# Paper should mention this rounding.

# --- T0param examples tried ---
# T0param_old <- c(lambda = 0.1, beta_X1 = 0.5, beta_X2 = 0.3)
# T0param <- c(lambda = exp(-6), beta_X1 = 0.5, beta_X2 = 0.5, nu = 2)

# --- Censoring rate notes ---
# WANT: censoring_rate = P(A+C<T) ~0.2
# censoring_rate <- sum(A + C < T) / N  # was ~0.219 with old params
# With new threshold~Unif(0,20): censoring ~26%

# --- Truncation probability notes ---
# WANT: truncation_prob = P(T0<A0), sum(T0<=A0)/N0 ~ 0.25-0.40
# truncation_prob = sum(T0 <= A0) / N0
# Old values: 0.4086 with Weibull + A0~Unif(0,17.5)

# --- Y(s) model ---
# Y(s) = mu_j(s,t;θ) * exp(beta'X) * y_err
# mu_j(s,t;θ) = g_j(t) * r_j(s,t;θ)
# y_err ~ Gamma(shape=10, rate=10), mean=1, var=0.1

# --- Y(s) old scenario designs (before current sce1/sce2) ---
# 1.gamma: rj(s,t;θ) = fj(s;0)/Fj(t;0), gj0(t) = 1
#   true theta: cause1:2, cause2:5
# 2.sigmoid: rj(s,t;θ) = 1/(1 + e^{θ(s-t/2)}), gj0(t)=1 or 1/(1+t)
#   true theta: cause1:1, cause2:2

# --- Current scenario designs (finalized 3/17/26, see rfun_final_0317.R) ---
# Scenario 1 (both DOWN):
#   j=1: s+s² (sce 2.2), theta=(-1,-1), c=0.5, g(t)=0.5+0.1t
#   j=2: (t-s)+(t-s)² (sce 1.2), theta=(0.5,1.0), c=0.5, g=0.5
#   mu0(s) = 5·exp(-ln(10)/30·s), 5 → 0.5 at s=30
# Scenario 2 (both UP):
#   j=1: Sigmoid (sce 2.1), theta=2, center=0, g=4
#   j=2: s+s*t centered (sce 1.1), theta=(0.5,0.5), c=0.5, g=4·exp(-0.05t)
#   mu0(s) = 0.5·exp(ln(6)/30·s), 0.5 → 3.0 at s=30
# Beta truth: all beta1=1, beta2=-1 for all groups (j=1, j=2, survivor)

# --- Unused code: baseline-only output ---
# df <- df_l[df_l$V == 0,]
# out <- list(); out$df <- df; out$df_l <- df_l; return(out)

# --- Output columns ---
# id: 1:N
# T: failure time
# A: truncation time (baseline time)
# S: measurement time (A<=S<=Z), S = A + V
# Z: observed survival time, Z = min(T, A+C)
# Pi: cause of failure: 1 or 2
# delta: censoring indicator: 1=event, 0=censored
# delPi: delta*Pi: uncensored cause of failure
# X1, X2: two covariates
# V: measurement point (time since enrollment)
# Y: marker at time S
