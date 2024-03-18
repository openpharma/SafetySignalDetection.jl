"""

Blinded Analysis Model

This Turing model is used to generate posterior samples of the adverse event probabilities
`pi_exp` in the experimental arm and `pi_ctrl` in the control arm given a blinded analysis
of a trial with `exp_proportion` ratio of experimental arm patients relative to all patients.

blinded_analysis_model(
    y::Vector{Bool}, 
    time::Vector{Float64}, 
    prior_exp::Distribution, 
    prior_ctrl::Distribution,
    exp_proportion::Float64)

"""
@model function blinded_analysis_model(
    y::Vector{Bool}, 
    time::Vector{Float64}, 
    prior_exp::Distribution, 
    prior_ctrl::Distribution,
    exp_proportion::Float64)

    n = length(y)
    pi_exp ~ prior_exp
    pi_ctrl ~ prior_ctrl
    mu_exp = log(-log(1 - pi_exp))
    mu_ctrl = log(-log(1 - pi_ctrl))

    for i in 1:n
        x_exp = mu_exp + log(time[i])
        x_ctrl = mu_ctrl + log(time[i])
        prob_exp = 1 - exp(-exp(x_exp))
        prob_ctrl = 1 - exp(-exp(x_ctrl))
        y[i] ~ MixtureModel(
            Bernoulli[
                Bernoulli(prob_exp),
                Bernoulli(prob_ctrl)
            ],
            [exp_proportion, 1 - exp_proportion]
        )
    end

end


"""

Blinded Analysis Posterior Samples Generation

This function wraps the Turing model `blinded_analysis_model` and runs it for a data frame `df` with:
    - `y`: `Bool` (did the adverse event occur?)
    - `time`: `Float64` (time until adverse event or until last treatment or follow up)

Note that arguments for the number of samples per chain and the number of chains have to be passed as well.
    
It returns a `DataFrame` with the posterior samples for `pi_exp` and `pi_ctrl`.
"""
function blinded_analysis_samples(
    df::DataFrame,
    prior_exp::Distribution, 
    prior_ctrl::Distribution, 
    exp_proportion::Float64,
    args...
    )
    chain = sample(
        blinded_analysis_model(df.y, df.time, prior_exp, prior_ctrl, exp_proportion),  
        NUTS(0.65),
        MCMCThreads(),
        args...
    )
    df = DataFrame(chain)
    select!(df, [:pi_exp, :pi_ctrl])
    
end