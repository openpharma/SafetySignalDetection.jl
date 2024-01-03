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
