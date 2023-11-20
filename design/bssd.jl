
# Implement a Bayesian safety signal monitoring method in Turing
# 2023
# Kristian Brock, kristian.brock@gmail.com, 
# and Daniel Sabanes Bove, daniel.sabanes_bove@roche.com

using Turing
using Distributions
using StatsPlots
using DataFrames
using CSV
pwd()
df = DataFrame(CSV.File("design/bssd_current.csv"))

## Unblinded analysis ----------------------------------------------------------------------

# Fit BSSD model to *unblinded* interim data (i.e. we know tmt allocations)
@model function unblinded_bssd(y, time, tmt, prior1, prior2)

    n = length(y)
    pi1 ~ prior1
    pi2 ~ prior2
    mu1 = log(-log(1 - pi1))
    mu2 = log(-log(1 - pi2))

    for i in 1:n
        if tmt[i] == 1
            x = mu1 + log(time[i])
        else
            x = mu2 + log(time[i])
        end
        prob = 1 - exp(-exp(x))
        y[i] ~ Bernoulli(prob)
    end

end;

# Specify priors. Use robustified mixture on control arm:
prior1 = MixtureModel(
    Beta[
        Beta(57.9, 332.2),
        Beta(1, 1)
    ], 
    [0.9, 0.1]
)

# And diffuse prior on treatment arm:
prior2 = MixtureModel(
    Beta[
        Beta(1, 1)
    ], 
    [1]
)

fit_unblinded_bssd = sample(unblinded_bssd(df.y, df.time, df.tmt, prior1, prior2), 
             HMC(0.05, 10), MCMCThreads(), 1000, 4)

# Those pi1 and pi2 estimates are not far away from the empirical proportions in the data:
fit_unblinded_bssd
names(fit_unblinded_bssd)
mean(fit_unblinded_bssd[:pi1])
mean(fit_unblinded_bssd[:pi2])

df_tmt1 = filter(:tmt => ==(1), df)
mean(df_tmt1.y)

df_tmt2 = filter(:tmt => ==(2), df)
mean(df_tmt2.y)


# Prob(T > C | X) = Prob(C > T | X):
mean(DataFrame(fit_unblinded_bssd).pi1 .< DataFrame(fit_unblinded_bssd).pi2)
# ~0.99


## Meta-analytic prior model fitting ----------------------------------------------------------------------

# This will work similarly as above unblinded analysis.
# we basically need to sample from the posterior of the mean a and the discount factor b
# and then draw samples from the Beta predictive distribution to get samples from pi*. 
# - this might even work out of the box with the predict method

# new input: 
# - trialindex, giving the trial index (1, ..., n_trials) for each y and time element. 
# - prior_a, prior on a
# - prior_b, prior on b

@model function meta_analytic(y, time, trialindex, prior_a, prior_b)

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

# Let's just simulate some random data quickly to try this out.
using Random, Distributions
Random.seed!(123)
k = 5 # number of trials
m = 50 # number of patients per trial
df_history = DataFrame(
    y = rand(Bernoulli(0.2), k * m),
    time = rand(Exponential(1), k * m),
    trialindex = repeat(1:k, m)
)

# Now let's fit the MAP model.
meta_analytic

fit_meta_analytic = sample(
    meta_analytic(df_history.y, df_history.time, df_history.trialindex, Beta(2, 8), Beta(9, 10)), 
    HMC(0.05, 10), MCMCThreads(), 1000, 4
)
# It is important here that the priors on a and b are realistic, they cannot be completely
# uninformative. Otherwise initial values are not found and sampler does not return quickly.

# Now let's sample from the predictive distribution for a new control arm:
# This does not seem easy with the `predict` function because we are not looking at the outcome y here.
# So we just directly sample from Beta(a * b * n, (1 - a) * b * n) where we plug
# in posterior samples for a and b:

n = length(df_history.y)
fit_meta_analytic_samples = DataFrame(fit_meta_analytic)
Random.seed!(123)
pi_star_samples = [rand(Beta(row.a * row.b * n, (1 - row.a) * row.b * n),1)[1] for row in eachrow(fit_meta_analytic_samples)]

histogram(pi_star_samples)
# This can then be approximated with the beta mixture prior (see above) in the interim step.



## Mixture approximation ----------------------------------------------------------------------

# Pkg.add("ExpectationMaximization")
using ExpectationMaximization
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
histogram(y)
stephist(y, label = "beta mixture", norm = :pdf)

# Initial guess.
mix_guess = MixtureModel([Beta(1, 1), Beta(1, 1)], [0.5, 1 - 0.5])
test = rand(mix_guess, N)

# Fit the MLE with the stochastic EM algorithm:
# (note that the classic EM algorithm does not work, see https://github.com/dmetivie/ExpectationMaximization.jl/issues/9,
# but for our purposes that is not important)
mix_mle = fit_mle(mix_guess, y, method = StochasticEM())
plot!(x -> pdf(mix_mle, x), label = "fitted distribution")

function fitmix2(x)
	# Fit a two-compnent Beta mixture to numerical sample x 
	mix_guess = MixtureModel([Beta(1, 1), Beta(1, 1)], [0.5, 1 - 0.5])
	test = rand(mix_guess, N)
	# Fit the MLE with the stochastic EM algorithm:
	# (note that the classic EM algorithm does not work, see https://github.com/dmetivie/ExpectationMaximization.jl/issues/9,
	# but for our purposes that is not important)
	mix_mle = fit_mle(mix_guess, x, method = StochasticEM())
	mix_mle
end;

mix1 = fitmix2(pi_star_samples)
mix1


## Blinded analysis ----------------------------------------------------------------------

# Here we either need
# - introduce unobserved treatment indicator with the correct randomization distribution
#   - here https://turing.ml/dev/tutorials/04-hidden-markov-model/ could be good inspiration
#   - we need to be careful because we have discrete parameters then, see also 
#     https://turing.ml/dev/tutorials/1-gaussianmixturemodel/ for how to handle this
# - or work with custom likelihood contribution (but that can limit sampling downstream analysis)

# Fit BSSD model to *blinded* interim data (i.e. we do NOT know the treatment allocations, hence no `tmt` argument).
# Here the additional argument `experimental_proportion` is the proportion of patients randomized to the experimental
# arm in the blinded trial, e.g. if 2:1 randomization it is 1/3.
@model function blinded_bssd(y, time, prior_exp, prior_ctrl, experimental_proportion)

    n = length(y)
    pi_exp ~ prior_exp
    pi_ctrl ~ prior_ctrl
    mu_exp = log(-log(1 - pi_exp))
    mu_ctrl = log(-log(1 - pi_ctrl))
    is_exp = Vector{Int}(undef, n)

    for i in 1:n
        is_exp[i] ~ Bernoulli(experimental_proportion)
        if is_exp[i] == 1
            x = mu_exp + log(time[i])
        else
            x = mu_ctrl + log(time[i])
        end
        prob = 1 - exp(-exp(x))
        y[i] ~ Bernoulli(prob)
    end

end;

# Let's try to fit this now.

prior_ctrl = MixtureModel(
    Beta[
        Beta(57.9, 332.2),
        Beta(1, 1)
    ], 
    [0.9, 0.1]
)

prior_exp = MixtureModel(
    Beta[
        Beta(1, 1)
    ], 
    [1]
)

using FreqTables
freqtable(df.tmt)
# so here was 73 times no. 2 i.e. the experimental arm, and 27 times control.
# so this is what we use below as randomization ratio

# Need to see how many particles we need here... for now we just stay with 100 as in the website example.
# gibbs_sampler = Gibbs(PG(100, :is_exp), HMC(0.05, 10, :pi_exp, :pi_ctrl))
# fit_blinded_bssd = sample(blinded_bssd(df.y, df.time, prior_exp, prior_ctrl, 73/100), 
#              gibbs_sampler, MCMCThreads(), 1000, 4)
# Wall duration: 635 secs!
# mean(DataFrame(fit_blinded_bssd).pi_ctrl .< DataFrame(fit_blinded_bssd).pi_exp)
# ~0.86

# This seems very slow. Let's try with a mixture model for the likelihood, so integrating out the treatment assignment.
# https://discourse.julialang.org/t/mixture-model-fitting-with-turing/51706
@model function blinded_bssd_mix(y, time, prior_exp, prior_ctrl, experimental_proportion)

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
            [experimental_proportion, 1 - experimental_proportion]
        )
    end

end;

fit_bssd_mix = sample(blinded_bssd_mix(df.y, df.time, prior_exp, prior_ctrl, 73/100), 
             HMC(0.05, 10), MCMCThreads(), 1000, 4)
# Wall duration: < 2 secs! so more than 300 times faster than above version with class indicators.
# Also ESS is much higher here (700 and 200, vs. 100 and 50 for the version above).

mean(DataFrame(fit_bssd_mix).pi_ctrl .< DataFrame(fit_bssd_mix).pi_exp)             

# ~0.88 so close to the above, which is a nice confirmation. If we ran the above sampler
# longer then we would expect to get even closer.

# so this is not as high as in the unblinded analysis, which makes sense because we know less
# and therefore we are less sure about the difference between the two proportions.
