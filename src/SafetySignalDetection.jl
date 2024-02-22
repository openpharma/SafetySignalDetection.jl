module SafetySignalDetection

using Turing
using StatsPlots
using Distributions
using SpecialFunctions
using Statistics
using LinearAlgebra
using ExpectationMaximization

export 
    meta_analysis_model,
    meta_analytic_samples,
    fit_beta_mixture

include("meta_analysis.jl")
include("fit_mle.jl")
include("fit_beta_mixture.jl")

end 