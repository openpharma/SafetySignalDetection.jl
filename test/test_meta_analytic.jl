@testset "meta_analytic.jl" begin
    rng = StableRNG(123)

    n_trials = 5
    n_patients = 50
    df = DataFrame(
        y = rand(rng, Bernoulli(0.2), n_trials * n_patients),
        time = rand(rng, Exponential(1), n_trials * n_patients),
        trialindex = repeat(1:n_trials, n_patients)
    )

    chain = sample(
        rng,
        meta_analytic(df.y, df.time, df.trialindex, Beta(2, 8), Beta(9, 10)), 
        HMC(0.05, 10), 
        1000
    )

    check_numerical(chain, [:a], [0.223], rtol=0.001)
    check_numerical(chain, [:b], [0.485], rtol=0.001)
end

@testset "Reconcile meta_analytic on known datasets with rstan" begin
    # Create MAP priors on known datasets and compare to rstan output
    
    seed = 2024
    prior_a = Beta(1 / 3, 1 / 3)
    prior_b = Beta(5, 5)
    prob_ctrl = 0.333

    # Small historical dataset
    df_small = DataFrame(CSV.File("small_historic_dataset.csv"))
    df_small.y = map(x -> x == 1, df_small.y) # Cast to Vector{Bool}
    rng = StableRNG(123)
    map_small = sample(
        rng,
        meta_analytic(df_small.y, df_small.time, df_small.trial, 
                      prior_a, prior_b), 
        HMC(0.05, 10), 
        10_000
    )
    check_numerical(map_small, [:a], [0.17], atol = 0.01)
    check_numerical(map_small, [:b], [0.51], atol = 0.01)
    

    # Large historical dataset
    df_large = DataFrame(CSV.File("large_historic_dataset.csv"))
    df_large.y = map(x -> x == 1, df_large.y) # Cast to Vector{Bool}
    rng = StableRNG(123)
    map_large = sample(
        rng,
        meta_analytic(df_large.y, df_large.time, df_large.trial, 
                      prior_a, prior_b), 
        HMC(0.05, 10), 
        10_000
    )
    check_numerical(map_large, [:a], [0.13], atol = 0.01)
    check_numerical(map_large, [:b], [0.55], atol = 0.01)

end