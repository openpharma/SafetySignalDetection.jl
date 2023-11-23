using Distributions
import Distributions: fit_mle, varm
using SpecialFunctions
using LinearAlgebra

fit_mle(Beta, [0.7, 0.3, 0.99, 0.5, 0.9, 0.1, 0.01, 0.0001])

# Version 2 which operates on the log scale, i.e.:
# γ := log(α)
# δ := log(β)
# This should be safer in general because we can contain negative values during the Newton-Raphson iterations.
function fit_mle_logscale(::Type{<:Beta}, x::AbstractArray{T};
    maxiter::Int=1000, tol::Float64=1e-14) where T<:Real

    α₀,β₀ = params(fit(Beta, x))

    g₁ = mean(log.(x))
    g₂ = mean(log.(one(T) .- x))

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

fit_mle_logscale(Beta, [0.7, 0.3, 0.99, 0.5, 0.9, 0.1, 0.01, 0.0001])


# Now on to the weighted version

# sufficient statistics - these capture everything of the data we need
struct BetaStats <: SufficientStats
    sum_log_x::Float64 # (weighted) sum of log(x)
    sum_log_1mx::Float64 # (weighted) sum of log(1 - x)
    tw::Float64 # total sample weight
    x_bar::Float64 # (weighted) mean of x
    v_bar::Float64 # (weighted) variance of x
end

function suffstats(::Type{<:Beta}, x::AbstractArray{T}, w::AbstractArray{Float64}) where T<:Real
    
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

xtest = [0.7, 0.3, 0.99, 0.5, 0.9, 0.1, 0.01, 0.0001]
suffstats(Beta, xtest)


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
fit_mle(::Type{<:Beta}, x::AbstractArray{T}, w::AbstractArray{Float64}; maxiter::Int=1000, tol::Float64=1e-14) where T<:Real = fit_mle(Beta, suffstats(Beta, x, w))

# now let's try it out:

sstest = suffstats(Beta, xtest)
fit_mle(Beta, )
fit_mle(Beta, [0.7, 0.3, 0.99, 0.5, 0.9, 0.1, 0.01, 0.0001], ones(Float64, 8))