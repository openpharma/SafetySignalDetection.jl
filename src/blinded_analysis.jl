"""
    Blinded Analysis Model

This Turing model is used to generate posterior samples of the adverse event probabilities
`pi_exp` in the experimental arm and `pi_ctrl` in the control arm given a blinded analysis
of a trial with `exp_proportion` ratio of experimental arm patients relative to all patients.
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
