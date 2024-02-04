function init_beta_dists(n_components::Int) 
	zero_one_range = range(start = 0, stop = 1, length = n_components + 2)
	alpha_range = zero_one_range[2:(n_components + 1)]
	[Beta(alpha, 1) for alpha in alpha_range]
end

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
	
	# We initialize here with Beta distributions that are not identical but have increasing alpha parameters.
	beta_dists = init_beta_dists(n_components)
	mix_guess = MixtureModel(beta_dists)
	
	# Fit the MLE with the classic EM algorithm.
	fit_mle(mix_guess, x)
end
