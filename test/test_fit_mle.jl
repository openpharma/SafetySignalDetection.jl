@testset "fit_mle.jl" begin

    Random.seed!(1234)

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
    
    # Initial guess.
    mix_guess = MixtureModel([Beta(0.33, 1), Beta(0.66, 1)], [0.5, 1 - 0.5])
       
    # Fit the MLE with the classic EM algorithm:
    mix_mle = fit_mle(mix_guess, y)

     # Compare parameter estimates with true parameter values.
     @test mix_mle.components[1].α ≈ α₂ rtol = 0.1
     @test mix_mle.components[1].β ≈ β₂ rtol = 0.1
     @test mix_mle.components[2].α ≈ α₁ rtol = 0.1
     @test mix_mle.components[2].β ≈ β₁ rtol = 0.1
     @test mix_mle.prior.p[1] ≈ 1 - π rtol = 0.01
     @test mix_mle.prior.p[2] ≈ π rtol = 0.01
end