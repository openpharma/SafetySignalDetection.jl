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
