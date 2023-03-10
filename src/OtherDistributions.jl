# additional support for Gaussian and Bernoulli generative distributions 

function log_normal(x::AbstractMatrix{S}, mu::AbstractMatrix{S}, logsigma::AbstractVecOrMat{S}) where S <: Real
    return logpdf.(Normal.(mu, sqrt.(exp.(logsigma))), x)
end

function log_binary(x::AbstractMatrix{S}, dec_z::AbstractMatrix{S}, eps::S=S(1e-8)) where S <: Real
    # binary cross-entropy between reconstruction and observed data 
    return x .* log.(dec_z .+ eps(S)) .+ (one(S) .- x) .* log.(one(S) .- dec_z .+ eps(S))
end
