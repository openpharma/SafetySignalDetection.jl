"""
    Meta Analysis Model

This Turing model is used to generate posterior samples of the parameters `a` and `b`.
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

This function wraps the Turing model and runs it for a data frame `df` with:
    - `y`: Bool (did the adverse event occur?)
    - `time`: Float64 (time until adverse event or until last treatment or follow up)
    - `trialindex`: Int64 (index of trials, starting from 1 and consecutively numbered)

Note that arguments for the number of samples per chain and the number of chains have to be passed as well.
    
It returns an array with the samples from the meta analytic prior.
"""
function meta_analytic_samples(
    df::DataFrame,
    prior_a::Distribution, 
    prior_b::Distribution, 
    args...
    )
    chain = sample(
        meta_analysis_model(df.y, df.time, df.trialindex, prior_a, prior_b),  
        HMC(0.05, 10), 
        MCMCThreads(),
        args...
    )
    n = length(df.y)
    chain_df = DataFrame(chain)
    [rand(Beta(row.a * row.b * n, (1 - row.a) * row.b * n),1)[1] for row in eachrow(chain_df)]
    
end