# additional support for Gaussian and Bernoulli generative distributions 

# slow version 
#function log_normal(x::AbstractMatrix{S}, mu::AbstractMatrix{S}, logsigma::AbstractVecOrMat{S}) where S <: Real
#    return logpdf.(Normal.(mu, sqrt.(exp.(logsigma))), x)
#end

"""
    log_normal(x::AbstractMatrix{S}, μ::AbstractMatrix{S}, logσ::AbstractVecOrMat{S}) where S <: Real

Log likelihood (scalar) of a minibatch according to a Gaussian generative model.

# Arguments
- `x`: data
- `μ`: mean of the Gaussian distribution (shape: minibatch x vars)
- `logσ`: log standard deviation parameter (has to be positive support) (shape: minibatch x vars)
"""
function log_normal(x::AbstractMatrix{S}, μ::AbstractMatrix{S}, logσ::AbstractVecOrMat{S}) where S <: Real
    res = @fastmath (-(x .- μ).^2 ./ (2.0f0 .* exp.(logσ))) .- 0.5f0 .* (log(S(2π)) .+ logσ)
    return res
end

"""
    log_binary(x::AbstractMatrix{S}, dec_z::AbstractMatrix{S}) where S <: Real

Log likelihood (scalar) of a minibatch according to a Bernoulli generative model.

# Arguments
- `x`: data
- `dec_z`: decoder output - transformed to success probability of the Bernoulli distribution (shape: minibatch x vars)
"""
function log_binary(x::AbstractMatrix{S}, dec_z::AbstractMatrix{S}) where S <: Real
    # binary cross-entropy between reconstruction and observed data 
    dec_z = sigmoid.(log.(dec_z))
    return x .* log.(dec_z .+ eps(S)) .+ (one(S) .- x) .* log.(one(S) .- dec_z .+ eps(S))
end