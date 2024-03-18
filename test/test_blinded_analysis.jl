@testset "blinded_analysis_model works as expected" begin
    rng = StableRNG(123)

    n_per_arm = 50
    df = DataFrame(
        y = vcat(
            rand(rng, Bernoulli(0.2), n_per_arm),
            rand(rng, Bernoulli(0.5), n_per_arm)
        ),
        time = rand(rng, Exponential(1), 2 * n_per_arm)
    )

    chain = sample(
        rng,
        blinded_analysis_model(df.y, df.time, Beta(2, 8), Beta(9, 10), 0.5), 
        NUTS(0.65), 
        1000
    )

    check_numerical(chain, [:pi_exp], [0.091], rtol=0.001)
    check_numerical(chain, [:pi_ctrl], [0.659], rtol=0.001)
end

@testset "blinded_analysis_samples works as expected" begin
    rng = StableRNG(123)

    n_per_arm = 50
    df = DataFrame(
        y = vcat(
            rand(rng, Bernoulli(0.2), n_per_arm),
            rand(rng, Bernoulli(0.5), n_per_arm)
        ),
        time = rand(rng, Exponential(1), 2 * n_per_arm)
    )

    samples = blinded_analysis_samples(df, Beta(2, 8), Beta(9, 10), 0.5, 100, 1)

    @test typeof(samples) == DataFrame
    @test nrow(samples) == 100
    @test ncol(samples) == 2
    @test names(samples) == ["pi_exp", "pi_ctrl"]
end