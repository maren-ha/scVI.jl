#--------------------------------------------------------------------------
# NegativeBinomial functions 
#--------------------------------------------------------------------------

using SpecialFunctions # for loggamma

LogGammaTerms(x, theta) = @. loggamma(x + theta) - loggamma(theta) - loggamma(one(eltype(theta)) + x)

function log_zinb_positive(x::AbstractMatrix{S}, mu::AbstractMatrix{S}, theta::AbstractVecOrMat{S}, zi::AbstractMatrix{S}, eps::S=S(1e-8)) where S <: Real
    """
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

function log_nb_positive(x::AbstractMatrix{S}, mu::AbstractMatrix{S}, theta::AbstractVecOrMat{S}, eps::S=S(1e-8)) where S <: Real
    """
    Log likelihood (scalar) of a minibatch according to a nb model.

    Parameters
    ----------
    x: Data
    mu: mean of the negative binomial (has to be positive support) (shape: minibatch x vars)
    theta: inverse dispersion parameter (has to be positive support) (shape: minibatch x vars)
    eps: numerical stability constant
    """
    if length(size(theta)) == 1
        # do some shit 
    end
    log_theta_mu_eps = @fastmath log.(theta .+ mu .+ eps)
    res = @fastmath theta .* (log.(theta .+ eps) .- log_theta_mu_eps) .+ x .* (log.(mu .+ eps) .- log_theta_mu_eps) .+ loggamma.(x .+ theta) .- loggamma.(theta) .- loggamma.(x .+ 1)
    return res 
end

function log_poisson(x::AbstractMatrix{S}, mu::AbstractMatrix{S}, eps::S=S(1e-8)) where S <: Real
    """
    Log likelihood (scalar) of a minibatch according to a Poisson model.

    Parameters
    ----------
    x: Data
    mu: mean=variance of the Poisson distribution (has to be positive support) (shape: minibatch x vars)
    eps: numerical stability constant
    """
    return logpdf.(Poisson.(mu), x)
end

function _convert_mean_disp_to_counts_logits(mu, theta, eps=1e-6)
    """
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
    logits = log.(mu .+ eps) .- log.(theta .+ eps) 
    total_count = theta 
    return total_count, logits 
end

function _convert_counts_logits_to_mean_disp(total_count, logits)
    """
    NB parameterizations conversion.
    Parameters
    ----------
    total_count: Number of failures until the experiment is stopped.
    logits: success logits.
    Returns
    -------
    the mean and inverse overdispersion of the NB distribution.
    """
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

function decodersample(m::scVAE, z::AbstractMatrix, library::AbstractMatrix)
    """
    Sample from (zero-inflated) negative binomial distribution 
    parametrised by mu, theta and zi (logits of dropout parameter)
    adapted from here: https://github.com/YosefLab/scvi-tools/blob/f0a3ba6e11053069fd1857d2381083e5492fa8b8/scvi/distributions/_negative_binomial.py#L420
    """
    px_scale, theta, mu, zi_logits = generative(m, z, library)
    if m.gene_likelihood == :nb
        return rand(NegativeBinomial.(theta, theta ./ (theta .+ mu)), size(mu))
    elseif m.gene_likelihood == :zinb
        samp = rand.(NegativeBinomial.(theta, theta ./ (theta .+ mu)))
        zi_probs = logits_to_probs(zi_logits)
        is_zero = rand(Float32, size(mu)) .<= zi_probs
        samp[is_zero] .= 0.0
        return samp
    else
        error("Not implemented")
    end
end

#=
# to test: 
Random.seed!(42)
x = first(dataloader)
z, qz_m, qz_v, ql_m, ql_v, library = inference(m,x)
samp = decodersample(m, z, library)
=#