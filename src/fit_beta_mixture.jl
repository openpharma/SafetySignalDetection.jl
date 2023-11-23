"""
    Fit a beta mixture to a vector of prior samples

This function returns a beta mixture of `n_components` components approximating
the distribution of the sample vector `x`.
"""
function fit_beta_mixture(x::AbstractArray{T}, n_components::Int) where T<:Real
    0 < n_components || throw(DomainError(n_components, "there must be at least one component"))
    # Remove outliers to stabilize the fitting.
	lower_quant = quantile!(x, 0.001)
	upper_quant = quantile!(x, 0.999)
	x = filter(y -> y > lower_quant && y < upper_quant, x)
	# Fit a Beta mixture to the numerical sample x.
    beta_dists = repeat([Beta(1, 1)], n_components)
    init_weights = ones()
	mix_guess = MixtureModel(beta_dists)
	# Fit the MLE with the stochastic EM algorithm.
	# (Note that the classic EM algorithm does not work yet, see https://github.com/dmetivie/ExpectationMaximization.jl/issues/9.)
	fit_mle(mix_guess, x, method = StochasticEM())
end
