module BSSD

using Turing
using StatsPlots
using Distributions
using Statistics
using ExpectationMaximization

export 
    meta_analytic,
    fit_beta_mixture

include("meta_analytic.jl")
include("fit_beta_mixture.jl")

end
