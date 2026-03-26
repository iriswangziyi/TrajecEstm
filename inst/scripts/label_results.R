#!/usr/bin/env Rscript

# --------------------------------------------
# label_results.R – Label + summarize simulation output
# Run this locally after downloading all_results.rds
# --------------------------------------------

library(dplyr)
library(xtable)
library(tidyr)

# ======= Load Simulation Output =======
res_mat <- readRDS("all_results2.rds")

# ======= Construct Column Names =======
pt_names <- c("beta1_j1", "beta2_j1", "theta1_j1",
              "beta1_j2", "beta2_j2", "theta1_j2", "theta2_j2")

mu_names <- paste0("mu_j", rep(1:2, each = 15), "_", rep(1:15, 2))
se_names   <- paste0("SE_", pt_names)
covP_names <- paste0("CovP_", pt_names)
covN_names <- paste0("CovN_", pt_names)

colnames(res_mat) <- c(pt_names, mu_names, se_names, covP_names, covN_names)

# ======= Extract Sections =======
est_mat   <- res_mat[, pt_names]
mu_mat    <- res_mat[, mu_names]
se_mat    <- res_mat[, se_names]
covP_mat  <- res_mat[, covP_names]
covN_mat  <- res_mat[, covN_names]

# ======= True Values =======
truth <- c(1, -1, 2,   # j = 1
           1, -1, 1, -2)  # j = 2

# ======= Compute Summary Table =======
bias <- colMeans(est_mat) - truth
se   <- apply(est_mat, 2, sd)
ase  <- colMeans(se_mat)
cp_q <- colMeans(covP_mat) * 100
cp_n <- colMeans(covN_mat) * 100

df_summary <- data.frame(
    Parameter = pt_names,
    Truth     = truth,
    Bias      = round(bias, 3),
    SE        = round(se, 3),
    ASE       = round(ase, 3),
    CP_quantile = round(cp_q, 1),
    CP_normal  = round(cp_n, 1)
)

# ======= Output Summary Table =======
cat("=== Table: Point Estimation Summary ===\n")

print(
    xtable(df_summary, digits = c(0, 0, 2, 2, 2, 1, 1, 1)),
    include.rownames = FALSE,
    type = "latex")

#write.csv(df_summary, "summary_table.csv", row.names = FALSE)

# ======= Full mû Table for R use (no rounding) =======

# Load grids
grid_j1 <- readRDS("mu_grid_j1.rds")
grid_j2 <- readRDS("mu_grid_j2.rds")

# Assign mû estimates from mu_avg
grid_j1$mu <- mu_avg[1:15]
grid_j2$mu <- mu_avg[16:30]

# Combine full-precision version (for R use)
df_mu_wide <- data.frame(
    t = grid_j1$t,
    s = grid_j1$s,
    mu_j1 = grid_j1$mu,
    mu_j2 = grid_j2$mu
)

# Save full version for R/CSV use
#saveRDS(df_mu_wide, "mu_table_full.rds")
#write.csv(df_mu_wide, "mu_table_full.csv", row.names = FALSE)

# ======= Display version for LaTeX (rounded) =======

df_mu_disp <- df_mu_wide %>%
    mutate(
        t = round(t, 2),
        s = round(s, 2),
        mu_j1 = round(mu_j1, 3),
        mu_j2 = round(mu_j2, 3)
    )

# Rename columns with LaTeX math
colnames(df_mu_disp) <- c("t", "s", "$\\hat{\\mu}_1$", "$\\hat{\\mu}_2$")

# Print LaTeX table
cat("=== Table: Estimated mu(t, s) for j = 1, 2 ===\n")
print(
    xtable(df_mu_disp, digits = c(0, 2, 2, 3, 3)),
    include.rownames = FALSE,
    sanitize.colnames.function = identity,
    type = "latex"
)
