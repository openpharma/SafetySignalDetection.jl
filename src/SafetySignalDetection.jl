module SafetySignalDetection

using Turing
using StatsPlots
using DataFrames
using Distributions
using SpecialFunctions
using Statistics
using LinearAlgebra
using ExpectationMaximization

export 
    blinded_analysis_model,
    blinded_analysis_samples,
    meta_analysis_model,
    meta_analytic_samples,
    fit_beta_mixture

include("blinded_analysis.jl")
include("meta_analysis.jl")
include("fit_mle.jl")
include("fit_beta_mixture.jl")

end 