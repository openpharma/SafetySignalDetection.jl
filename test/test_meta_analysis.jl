@testset "Check that meta_analysis_model works as before" begin
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
        meta_analysis_model(df.y, df.time, df.trialindex, Beta(2, 8), Beta(9, 10)), 
        HMC(0.05, 10), 
        1000
    )

    check_numerical(chain, [:a], [0.223], rtol=0.001)
    check_numerical(chain, [:b], [0.485], rtol=0.001)
end

@testset "Reconcile meta_analysis_model with rstan on small historical dataset" begin
    # Create MAP priors on known small historical dataset and compare to rstan    
    
    prior_a = Beta(1 / 3, 1 / 3)
    prior_b = Beta(5, 5)
    prob_ctrl = 0.333

    df_small = DataFrame(CSV.File("small_historic_dataset.csv"))
    df_small.y = map(x -> x == 1, df_small.y) # Cast to Vector{Bool}
    rng = StableRNG(123)
    map_small = sample(
        rng,
        meta_analysis_model(df_small.y, df_small.time, df_small.trial, 
                            prior_a, prior_b), 
        NUTS(0.65),
        10_000
    )
    check_numerical(map_small, [:a], [0.17], rtol = 0.01)
    check_numerical(map_small, [:b], [0.51], rtol = 0.01)

end

@testset "Reconcile meta_analysis_model with rstan on large historical dataset" begin
    # Create MAP priors on known large historical dataset and compare to rstan

    prior_a = Beta(1 / 3, 1 / 3)
    prior_b = Beta(5, 5)
    prob_ctrl = 0.333

    df_large = DataFrame(CSV.File("large_historic_dataset.csv"))
    df_large.y = map(x -> x == 1, df_large.y) # Cast to Vector{Bool}
    rng = StableRNG(123)
    map_large = sample(
        rng,
        meta_analysis_model(df_large.y, df_large.time, df_large.trial, 
                            prior_a, prior_b), 
        NUTS(0.65), 
        10_000
    )
    check_numerical(map_large, [:a], [0.13], rtol = 0.01)
    check_numerical(map_large, [:b], [0.55], rtol = 0.01)

end

@testset "meta_analytic_samples" begin
    rng = StableRNG(123)

    n_trials = 5
    n_patients = 50
    df = DataFrame(
        y = rand(rng, Bernoulli(0.2), n_trials * n_patients),
        time = rand(rng, Exponential(1), n_trials * n_patients),
        trialindex = repeat(1:n_trials, n_patients)
    )

    samples = meta_analytic_samples(df, Beta(2, 8), Beta(9, 10), 100, 1)

    @test typeof(samples) == Vector{Float64}
    @test length(samples) == 100
end
