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
