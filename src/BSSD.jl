module BSSD

using Turing
using StatsPlots
using Distributions
using Statistics

export 
    meta_analytic

include("meta_analytic.jl")
include("fit_beta_mixture.jl")

end
