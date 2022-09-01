#--------------------------------------------------------------------------
# NegativeBinomial functions 
#--------------------------------------------------------------------------

using SpecialFunctions # for loggamma

LogGammaTerms(x, theta) = @. loggamma(x + theta) - loggamma(theta) - loggamma(one(eltype(theta)) + x)

"""
    log_zinb_positive(x::AbstractMatrix{S}, mu::AbstractMatrix{S}, theta::AbstractVecOrMat{S}, zi::AbstractMatrix{S}, eps::S=S(1e-8)) where S <: Real

Log likelihood (scalar) of a minibatch according to a zinb model.

Parameters
----------
x: Data
mu: mean of the negative binomial (has to be positive support) (shape: minibatch x vars)
theta: inverse dispersion parameter (has to be positive support) (shape: minibatch x vars)
pi: logit of the dropout parameter (real support) (shape: minibatch x vars)
eps: numerical stability constant

Notes
-----
We parametrize the bernoulli using the logits, hence the softplus functions appearing.
"""
function log_zinb_positive(x::AbstractMatrix{S}, mu::AbstractMatrix{S}, theta::AbstractVecOrMat{S}, zi::AbstractMatrix{S}, eps::S=S(1e-8)) where S <: Real
    softplus_zi = @fastmath softplus.(-zi);    
    log_thetha_mu_eps = @fastmath log.(theta .+ mu .+ eps)
    GammaTerms = @fastmath LogGammaTerms(x, theta)
    @fastmath @. (
       (x .< eps) .* (softplus.(-zi .+ theta .* (log.(theta .+ eps) .- log_thetha_mu_eps)) .- softplus_zi)
    .+ (x .> eps) .* (-softplus_zi .- zi .+ theta .* (log.(theta .+ eps) .- log_thetha_mu_eps)
                    .+ x .* (log.(mu .+ eps) .- log_thetha_mu_eps) .+ GammaTerms)
    )
end

function log_nb_positive_julia(x, mu, theta)
    r = theta 
    p = theta ./ (theta .+ mu)
    return logpdf.(NegativeBinomial.(r, p), x)
end

"""
    log_nb_positive(x::AbstractMatrix{S}, mu::AbstractMatrix{S}, theta::AbstractVecOrMat{S}, eps::S=S(1e-8)) where S <: Real

Log likelihood (scalar) of a minibatch according to a nb model.

Parameters
----------
x: Data
mu: mean of the negative binomial (has to be positive support) (shape: minibatch x vars)
theta: inverse dispersion parameter (has to be positive support) (shape: minibatch x vars)
eps: numerical stability constant
"""
function log_nb_positive(x::AbstractMatrix{S}, mu::AbstractMatrix{S}, theta::AbstractVecOrMat{S}, eps::S=S(1e-8)) where S <: Real
    if length(size(theta)) == 1
        # do some shit 
    end
    log_theta_mu_eps = @fastmath log.(theta .+ mu .+ eps)
    res = @fastmath theta .* (log.(theta .+ eps) .- log_theta_mu_eps) .+ x .* (log.(mu .+ eps) .- log_theta_mu_eps) .+ loggamma.(x .+ theta) .- loggamma.(theta) .- loggamma.(x .+ 1)
    return res 
end

"""
    log_poisson(x::AbstractMatrix{S}, mu::AbstractMatrix{S}, eps::S=S(1e-8)) where S <: Real

Log likelihood (scalar) of a minibatch according to a Poisson model.

Parameters
----------
x: Data
mu: mean=variance of the Poisson distribution (has to be positive support) (shape: minibatch x vars)
eps: numerical stability constant
"""
function log_poisson(x::AbstractMatrix{S}, mu::AbstractMatrix{S}, eps::S=S(1e-8)) where S <: Real
    return logpdf.(Poisson.(mu), x)
end

"""
    _convert_mean_disp_to_counts_logits(mu, theta, eps=1e-6)

NB parameterizations conversion.
Parameters
----------
mu: mean of the NB distribution.
theta: inverse overdispersion.
eps: constant used for numerical log stability. (Default value = 1e-6)
Returns
-------
the number of failures until the experiment is stopped and the success probability.
"""
function _convert_mean_disp_to_counts_logits(mu, theta, eps=1e-6)
    logits = log.(mu .+ eps) .- log.(theta .+ eps) 
    total_count = theta 
    return total_count, logits 
end

"""
    _convert_counts_logits_to_mean_disp(total_count, logits)

NB parameterizations conversion.
Parameters
----------
total_count: Number of failures until the experiment is stopped.
logits: success logits.
Returns
-------
the mean and inverse overdispersion of the NB distribution.
"""
function _convert_counts_logits_to_mean_disp(total_count, logits)
    theta = total_count
    mu = exp.(logits) .* theta
    return mu, theta
end

function logits_to_probs(zi_logits)
    probs = sigmoid.(zi_logits)
    return probs
end

function _gamma(theta, mu)
    concentration = theta 
    rate = theta ./ mu
    scale = mu ./ theta
    # Important remark: Gamma is parametrized by the rate = 1/scale in Python 
    # but Gamma(α, θ)      # Gamma distribution with shape α and scale θ in Julia
    gamma_d = Gamma(concentration, scale)
    return gamma_d
end