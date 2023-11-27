module BSSD

using Turing
using StatsPlots
using Distributions
using Statistics
using ExpectationMaximization
using DataFrames

export 
    meta_analysis_model,
    meta_analytic_samples,
    fit_beta_mixture,
    blinded_analysis_model

include("meta_analysis.jl")
include("fit_beta_mixture.jl")
include("blinded_analysis.jl")

end
