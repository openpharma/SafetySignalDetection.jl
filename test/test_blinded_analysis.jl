@testset "blinded_analysis_model" begin
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
        HMC(0.05, 10), 
        1000
    )

    check_numerical(chain, [:pi_exp], [0.095], rtol=0.001)
    check_numerical(chain, [:pi_ctrl], [0.655], rtol=0.001)
end
