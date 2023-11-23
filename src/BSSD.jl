module BSSD

using Turing
using StatsPlots
using Distributions
using Statistics
using ExpectationMaximization
using DataFrames

export 
    meta_analytic_model,
    meta_analytic_samples,
    fit_beta_mixture

include("meta_analytic.jl")
include("fit_beta_mixture.jl")

end
