using Test
using StableRNGs
using Random
using Distributions
using BSSD

# Helper function for numerical tests.
# Taken from https://github.com/TuringLang/Turing.jl/blob/master/test/test_utils/numerical_tests.jl#L41 for now.
function check_numerical(chain,
    symbols::Vector,
    exact_vals::Vector;
    atol=0.2,
    rtol=0.0)
    for (sym, val) in zip(symbols, exact_vals)
        E = val isa Real ?
            mean(chain[sym]) :
            vec(mean(chain[sym], dims=1))
        @info (symbol=sym, exact=val, evaluated=E)
        @test E ≈ val atol=atol rtol=rtol
    end
end

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

    check_numerical(chain, )
end
