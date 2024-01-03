"""

Meta Analytic Prior Model

This Turing model is used to generate posterior samples of the parameters `a` and `b`.

    meta_analytic(
    y::Vector{Bool}, 
    time::Vector{Float64}, 
    trialindex::Vector{Int64}, 
    prior_a::Distribution, 
    prior_b::Distribution)

"""
@model function meta_analytic(
    y::Vector{Bool}, 
    time::Vector{Float64}, 
    trialindex::Vector{Int64}, 
    prior_a::Distribution, 
    prior_b::Distribution)

    n = length(y)
    n_trials = maximum(trialindex)

    a ~ prior_a
    b ~ prior_b
    pis ~ filldist(Beta(a * b * n, (1 - a) * b * n), n_trials)

    for i in 1:n
        pi = pis[trialindex[i]]
        mu = log(-log(1 - pi))
        x = mu + log(time[i])
        prob = 1 - exp(-exp(x))
        y[i] ~ Bernoulli(prob)
    end

end;