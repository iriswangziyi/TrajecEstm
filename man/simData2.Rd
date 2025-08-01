% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/dataGen.R
\name{simData2}
\alias{simData2}
\title{Generate Simulated Data With Time-Dependent X}
\usage{
simData2(
  T0param = c(lambda = 0.1, beta_X1 = 0.5, beta_X2 = 0.3),
  beta0 = c(beta11 = 1, beta12 = -1, beta21 = 1, beta22 = -1),
  N = 500,
  scenario = 2
)
}
\arguments{
\item{n}{Integer. Number of subjects to simulate.}

\item{p}{Integer. Dimension of the covariate vector \code{X}.}

\item{m}{Integer. Number of follow-up visits after baseline.}

\item{beta}{Numeric vector of length \code{p}. True regression coefficients.}

\item{theta}{Numeric. True value of \eqn{\theta}.}

\item{seed}{Integer (optional). RNG seed for reproducibility.}
}
\value{
A list with elements
\describe{
\item{\code{X}}{p × n design matrix (columns = subjects).}
\item{\code{A}}{Vector of left-truncation times (length n).}
\item{\code{V}}{List of visit-offset vectors, one per subject.}
\item{\code{Z}}{Vector of failure/censoring times (length n).}
\item{\code{Y}}{Vector of marker values at each recorded visit.}
\item{\code{Pi}}{Failure-type indicator (length n).}
\item{\code{Delta}}{Censoring indicator (length n).}
}
}
\description{
Creates a synthetic data set suitable for testing the trajectory
estimator.  Each subject enters the study at left-truncation time
\eqn{A_0}, is observed at baseline and at \code{m} follow-up visits,
and may experience a type-\code{j} failure or censoring at time
\eqn{Z}.  The marker trajectory is generated under Model (I) with
user-specified parameters.
}
\examples{
demo <- simData2(n = 100, p = 3, m = 2,
                 beta = c(0.3, -0.5, 1.0),
                 theta = 0.8, seed = 1)
str(demo)

}
