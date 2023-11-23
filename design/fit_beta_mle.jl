using Distributions
import Distributions: fit_mle, varm
using SpecialFunctions
using LinearAlgebra

# sufficient statistics - these capture everything of the data we need
struct BetaStats <: SufficientStats
    sum_log_x::Float64 # (weighted) sum of log(x)
    sum_log_1mx::Float64 # (weighted) sum of log(1 - x)
    tw::Float64 # total sample weight
    x_bar::Float64 # (weighted) mean of x
    v_bar::Float64 # (weighted) variance of x
end

function suffstats(::Type{<:Beta}, x::AbstractArray{T}, w::AbstractArray{T}) where T<:Real

    tw = 0.0
    sum_log_x = 0.0 * zero(T)
    sum_log_1mx = 0.0 * zero(T)
    x_bar = 0.0 * zero(T)

    for i in eachindex(x, w)
        @inbounds xi = x[i]
        0 < xi < 1 || throw(DomainError(xi, "samples must be larger than 0 and smaller than 1"))
        @inbounds wi = w[i]
        wi >= 0 || throw(DomainError(wi, "weights must be non-negative"))
        isfinite(wi) || throw(DomainError(wi, "weights must be finite"))
        @inbounds sum_log_x += wi * log(xi)
        @inbounds sum_log_1mx += wi * log(one(T) - xi)
        @inbounds x_bar += wi * xi
        tw += wi
    end
    sum_log_x /= tw   
    sum_log_1mx /= tw
    x_bar /= tw
    v_bar = varm(x, x_bar)

    BetaStats(sum_log_x, sum_log_1mx, tw, x_bar, v_bar)
end

# without weights we just put weight 1 for each observation
function suffstats(::Type{<:Beta}, x::AbstractArray{T}) where T<:Real
    
    w = ones(Float64, length(x))
    suffstats(Beta, x, w)

end

# generic fit function based on the sufficient statistics, on the log scale to be robust
function fit_mle(::Type{<:Beta}, ss::BetaStats;
    maxiter::Int=1000, tol::Float64=1e-14)

    # Initialization of parameters at the moment estimators (I guess)
    temp = ((ss.x_bar * (1 - ss.x_bar)) / ss.v_bar) - 1
    α₀ = ss.x_bar * temp
    β₀ = (1 - ss.x_bar) * temp

    g₁ = ss.sum_log_x
    g₂ = ss.sum_log_1mx

    θ= [log(α₀) ; log(β₀)]

    converged = false
    t=0
    while !converged && t < maxiter
        t += 1
        α = exp(θ[1])
        β = exp(θ[2])
        temp1 = digamma(α + β)
        temp2 = trigamma(α + β)
        temp3 = g₁ + temp1 - digamma(α)
        grad = [temp3 * α
                (temp1 + g₂ - digamma(β)) * β]
        hess = [((temp2 - trigamma(α)) * α + temp3) * α             temp2 * β * α
                temp2 * α * β       ((temp2 - trigamma(β)) * β + temp1 + g₂ - digamma(β)) * β  ]
        Δθ = hess\grad #newton step
        θ .-= Δθ
        converged = dot(Δθ,Δθ) < 2*tol #stopping criterion
    end

    α = exp(θ[1])
    β = exp(θ[2])
    return Beta(α, β)
end

# then define methods for the original data
fit_mle(::Type{<:Beta}, x::AbstractArray{T}; maxiter::Int=1000, tol::Float64=1e-14) where T<:Real = fit_mle(Beta, suffstats(Beta, x))
fit_mle(::Type{<:Beta}, x::AbstractArray{T}, w::AbstractArray{T}; maxiter::Int=1000, tol::Float64=1e-14) where T<:Real = fit_mle(Beta, suffstats(Beta, x, w))

# now let's try it out:

fit_mle(Beta, xtest)
fit_mle(Beta, xtest, ones(Float64, 8))
Distributions.fit(Beta, xtest)
wtest = rand(Uniform(), 8)
fit_mle(Beta, xtest, wtest)

# Now finally having this in place the classic EM algorithm should work:
using ExpectationMaximization
using Distributions
using Plots
using Random

Random.seed!(1234)

# Try to fit a beta mixture.
N = 50_000
α₁ = 10
β₁ = 5
α₂ = 5
β₂ = 10
π = 0.3

# Mixture Model of two betas.
mix_true = MixtureModel([Beta(α₁, β₁), Beta(α₂, β₂)], [π, 1 - π]) 

# Generate N samples from the mixture.
y = rand(mix_true, N)
stephist(y, label = "true distribution", norm = :pdf)

# Initial guess.
mix_guess = MixtureModel([Beta(1, 1), Beta(1, 1)], [0.5, 1 - 0.5])
test = rand(mix_guess, N)
stephist!(test, label = "guess distribution", norm = :pdf)

# Fit the MLE with the classic EM algorithm:
mix_mle = fit_mle(mix_guess, y)
fit_y = rand(mix_mle, N)
stephist!(fit_y, label = "fitMLE distribution", norm = :pdf)
# This does not seem to work? Why?

# Test first if we can downweight large values and get expected result.
test_weights = Float64.(ifelse.(y .> 0.5, 0.01, 1))
low_res = fit_mle(Beta, y, test_weights)

fit_low_res = rand(low_res, N)
stephist!(fit_low_res, label = "fit_low_res", norm = :pdf)

# Same for high values
test_weights2 = Float64.(ifelse.(y .< 0.5, 0.01, 1))
high_res = fit_mle(Beta, y, test_weights2)

fit_high_res = rand(high_res, N)
stephist!(fit_high_res, label = "fit_high_res", norm = :pdf)
# So that seems to work fine.