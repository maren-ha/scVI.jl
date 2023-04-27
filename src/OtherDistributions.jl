# additional support for Gaussian and Bernoulli generative distributions 

# slow version 
#function log_normal(x::AbstractMatrix{S}, mu::AbstractMatrix{S}, logsigma::AbstractVecOrMat{S}) where S <: Real
#    return logpdf.(Normal.(mu, sqrt.(exp.(logsigma))), x)
#end

function log_normal(x::AbstractMatrix{S}, μ::AbstractMatrix{S}, logσ::AbstractVecOrMat{S}) where S <: Real
    res = @fastmath (-(x .- μ).^2 ./ (2.0f0 .* exp.(logσ))) .- 0.5f0 .* (log(S(2π)) .+ logσ)
    return res
end

function log_binary(x::AbstractMatrix{S}, dec_z::AbstractMatrix{S}) where S <: Real
    # binary cross-entropy between reconstruction and observed data 
    return x .* log.(dec_z .+ eps(S)) .+ (one(S) .- x) .* log.(one(S) .- dec_z .+ eps(S))
end