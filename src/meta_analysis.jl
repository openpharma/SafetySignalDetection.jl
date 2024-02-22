"""

Meta Analysis Model

This Turing model is used to generate posterior samples of the parameters `a` and `b`.

meta_analysis_model(
    y::Vector{Bool}, 
    time::Vector{Float64}, 
    trialindex::Vector{Int64}, 
    prior_a::Distribution, 
    prior_b::Distribution)

"""
@model function meta_analysis_model(
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


"""
Meta Analytic Prior Samples Generation

This function wraps the Turing model `meta_analysis_model` and runs it for a data frame `df` with:
    - `y`: Bool (did the adverse event occur?)
    - `time`: Float64 (time until adverse event or until last treatment or follow up)
    - `trialindex`: Int64 (index of trials, starting from 1 and consecutively numbered)

    meta_analytic_samples(
    df::DataFrame,
    prior_a::Distribution, 
    prior_b::Distribution, 
    args...
    )

Note that arguments for the number of samples per chain and the number of chains have to be passed as well.
    
It returns an array with the samples from the meta analytic prior (MAP).
"""
function meta_analytic_samples(
    df::DataFrame,
    prior_a::Distribution, 
    prior_b::Distribution, 
    args...
    )
    chain = sample(
        meta_analysis_model(df.y, df.time, df.trialindex, prior_a, prior_b),  
        NUTS(0.65), 
        MCMCThreads(),
        args...
    )
    n_trials = maximum(df.trialindex)
    predictive_index = n_trials + 1
    pi_star_name = "pis[" * string(predictive_index) * "]"
    vec(chain[pi_star_name].data)
    
end
