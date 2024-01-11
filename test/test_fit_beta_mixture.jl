@testset "init_beta_dists(1)" begin
    result = SafetySignalDetection.init_beta_dists(1)
    @test typeof(result) <: Array
    @test length(result) == 1
    @test typeof(result[1]) <: Beta
    @test result[1].α ≈ 0.5
    @test result[1].β == 1
end

@testset "init_beta_dists(4)" begin
    result = SafetySignalDetection.init_beta_dists(4)
    @test typeof(result) <: Array
    @test length(result) == 4
    @test result[1].α ≈ 0.2
    @test result[1].β == 1
    @test result[2].α ≈ 0.4
    @test result[2].β == 1
    @test result[3].α ≈ 0.6
    @test result[3].β == 1
    @test result[4].α ≈ 0.8
    @test result[4].β == 1
end

@testset "fit_beta_mixture" begin

    Random.seed!(123)

    N = 100_000
    α₁ = 10
    β₁ = 5
    α₂ = 5
    β₂ = 10
    π = 0.6

    # Draw large sample from true beta mixture model.
    mix_true = MixtureModel([Beta(α₁, β₁), Beta(α₂, β₂)], [π, 1 - π])
    x = rand(mix_true, N)

    # Fit the beta mixture.
    mix_fit = fit_beta_mixture(x, 2)
    
    # Compare parameter estimates with true parameter values.
    @test mix_fit.components[1].α ≈ α₂ rtol = 0.1
    @test mix_fit.components[1].β ≈ β₂ rtol = 0.1
    @test mix_fit.components[2].α ≈ α₁ rtol = 0.1
    @test mix_fit.components[2].β ≈ β₁ rtol = 0.1
    @test mix_fit.prior.p[1] ≈ 1 - π rtol = 0.01
    @test mix_fit.prior.p[2] ≈ π rtol = 0.01
end

@testset "Reconcile fit_beta_mixture on small historical dataset" begin
    # Create mixture from MAP prior on known small historical dataset
    
    prior_a = Beta(1 / 3, 1 / 3)
    prior_b = Beta(5, 5)
    prob_ctrl = 0.333

    df_small = DataFrame(CSV.File("small_historic_dataset.csv"))
    df_small.y = map(x -> x == 1, df_small.y) # Cast to Vector{Bool}
    rng = StableRNG(123)
    map_small = sample(
        rng,
        meta_analytic(df_small.y, df_small.time, df_small.trial, 
                      prior_a, prior_b), 
        NUTS(0.65), 
        10_000
    )
    df_map_small = DataFrame(map_small)
    pi_star_alpha = df_map_small.a .* df_map_small.b * nrow(df_small)
    pi_star_beta = (1 .- df_map_small.a) .* df_map_small.b * nrow(df_small)
    pi_star = [rand(Beta(alpha, beta), 1) 
        for (alpha, beta) in zip(pi_star_alpha, pi_star_beta)]
    pi_star = collect(Iterators.flatten(pi_star)) # Flatten
    mix_small = fit_beta_mixture(pi_star, 2)
    @test mean(mix_small) ≈ mean(pi_star) rtol = 0.01
    @test std(mix_small) ≈ std(pi_star) rtol = 0.01

end

@testset "Reconcile fit_beta_mixture on large historical dataset" begin
    # Create mixture from MAP prior on known large historical dataset
    
    prior_a = Beta(1 / 3, 1 / 3)
    prior_b = Beta(5, 5)
    prob_ctrl = 0.333

    df_large = DataFrame(CSV.File("large_historic_dataset.csv"))
    df_large.y = map(x -> x == 1, df_large.y) # Cast to Vector{Bool}

    rng = StableRNG(123)
    map_large = sample(
        rng,
        meta_analytic(df_large.y, df_large.time, df_large.trial, 
                      prior_a, prior_b), 
        NUTS(0.65), 
        10_000
    )
    df_map_large = DataFrame(map_large)
    pi_star_alpha = df_map_large.a .* df_map_large.b * nrow(df_large)
    pi_star_beta = (1 .- df_map_large.a) .* df_map_large.b * nrow(df_large)
    pi_star = [rand(Beta(alpha, beta), 1) 
        for (alpha, beta) in zip(pi_star_alpha, pi_star_beta)]
    pi_star = collect(Iterators.flatten(pi_star)) # Flatten
    mix_large = fit_beta_mixture(pi_star, 2)
    @test mean(mix_large) ≈ mean(pi_star) rtol = 0.01
    @test std(mix_large) ≈ std(pi_star) rtol = 0.01

end
