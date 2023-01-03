#using Printf: @sprintf
rescale(A; dims=1) = (A .- mean(A, dims=dims)) ./ max.(std(A, dims=dims), eps())

"""
Compute the point perplexities `P` given its squared distances to the other points `D`
and the precision of Gaussian distribution `beta`.
"""
function Hbeta!(P::AbstractVector, D::AbstractVector, beta::Number)
    @inbounds P .= exp.(-beta .* D)
    sumP = sum(P)
    @assert (isfinite(sumP) && sumP > 0.0) "Degenerated P: sum=$sumP, beta=$beta"
    H = log(sumP) + beta * dot(D, P) / sumP
    @assert isfinite(H) "Degenerated H"
    @inbounds P .*= 1/sumP
    return H
end

"""
    perplexities(D::AbstractMatrix, tol::Number = 1e-5, perplexity::Number = 30.0;
                 [keyword arguments])
Convert `n×n` squared distances matrix `D` into `n×n` perplexities matrix `P`.
Performs a binary search to get P-values in such a way that each conditional
Gaussian has the same perplexity.
"""
function perplexities(D::AbstractMatrix{T}, tol::Number = 1e-5, perplexity::Number = 30.0;
                      max_iter::Integer = 50,
                      verbose::Bool=false) where T<:Number
    if !(issymmetric(D) && all(x -> x >= 0, D) )
        error("Distance matrix D must be symmetric and positive")
    end

    # initialize
    n = size(D, 1)
    P = fill(zero(T), n, n) # perplexities matrix
    beta = fill(one(T), n)  # vector of Normal distribution precisions for each point
    Htarget = log(perplexity) # the expected entropy
    Di = fill(zero(T), n)
    Pcol = fill(zero(T), n)

    # Loop over all datapoints
    for i in 1:n
        # Compute the Gaussian kernel and entropy for the current precision
        betai = 1.0
        betamin = 0.0
        betamax = Inf

        copyto!(Di, view(D, :, i))
        Di[i] = prevfloat(Inf) # exclude D[i,i] from minimum(), yet make it finite and exp(-D[i,i])==0.0
        minD = minimum(Di) # distance of i-th point to its closest neighbour
        @inbounds Di .-= minD # entropy is invariant to offsetting Di, which helps to avoid overflow

        H = Hbeta!(Pcol, Di, betai)
        Hdiff = H - Htarget

        # Evaluate whether the perplexity is within tolerance
        tries = 0
        while abs(Hdiff) > tol && tries < max_iter
            # If not, increase or decrease precision
            if Hdiff > 0.0
                betamin = betai
                betai = isfinite(betamax) ? (betai + betamax)/2 : betai*2
            else
                betamax = betai
                betai = (betai + betamin)/2
            end

            # Recompute the values
            H = Hbeta!(Pcol, Di, betai)
            Hdiff = H - Htarget
            tries += 1
        end
        verbose && abs(Hdiff) > tol && warn("P[$i]: perplexity error is above tolerance: $(Hdiff)")
        # Set the final column of P
        @assert Pcol[i] == 0.0 "Diagonal probability P[$i,$i]=$(Pcol[i]) not zero"
        @inbounds P[:, i] .= Pcol
        beta[i] = betai
    end
    # Return final P-matrix
    verbose && @info(mean(sqrt.(1 ./ beta)))
    return P, beta
end

function compute_transition_probs(X, perplexity::Number=30) # X: shape obs x vars
    X = X * (1.0/std(X)::eltype(X))
    D = pairwise(SqEuclidean(), X)
    P, beta = perplexities(D, perplexity)
    return Float32.(P)
end

function tsne_repel(z::AbstractMatrix{S}, P::AbstractMatrix{S}) where S <: Real 
    # S = eltype(z)
    latent_dim, batchsize = S.(size(z))
    nu = latent_dim - one(S)
    sum_y = vec(sum(z.^2, dims=1))
    num = SliceMap.mapcols(x -> x + sum_y, -2.0f0 .* (z'*z))'
    num = SliceMap.mapcols(x -> x + sum_y, num)
    num = num ./ nu

    p = P .+ (0.1f0/size(z,2))
    sum_p = vec(sum(p, dims=2))
    p = SliceMap.maprows(x -> x ./ sum_p, p)
    num = (1.0f0 .+ num).^(-(0.5f0*(nu .+ 1.0f0)))

    attraction = -sum(p .* log.(num .+ eps(S)))
    repellant = sum(log.(sum(num, dims=2).- 1.0f0))
    return (repellant + attraction) ./ batchsize
end

# with differentiation through P 
function scvis_loss(m::scVAE, x::AbstractMatrix{S}, P::Nothing=nothing; kl_weight::Float32=1.0f0, epoch::Int=1) where S <: Real
    encoder_input = m.log_variational ? log.(one(S) .+ x) : x
    q = m.z_encoder.encoder(x)
    qz_m = m.z_encoder.mean_encoder(q)
    qz_v = m.z_encoder.var_activation.(m.z_encoder.var_encoder(q)) .+ m.z_encoder.var_eps
    z = m.z_encoder.z_transformation(scVI.reparameterize_gaussian(qz_m, qz_v))
    ql_m, ql_v = nothing, nothing
    library = scVI.get_library(m, x, encoder_input)
    # z, qz_m, qz_v, ql_m, ql_v, library = scVI.inference(m, x)
    # size(z) = latent_dim x batchsize
    px_scale, px_r, px_rate, px_dropout = scVI.generative(m, z, library)
    kl_divergence_z = -0.5f0 .* sum(1.0f0 .+ log.(qz_v) - qz_m.^2 .- qz_v, dims=1) # 2 

    if !m.use_observed_lib_size # TODO!
        local_library_log_means,local_library_log_vars = _compute_local_library_params() 
        kl_divergence_l = kl(Normal(ql_m, ql_v.sqrt()),Normal(local_library_log_means, local_library_log_vars.sqrt())).sum(dim=1)
    else
        kl_divergence_l = 0.0f0
    end

    reconst_loss = scVI.get_reconstruction_loss(m, x, px_rate, px_r, px_dropout)
    kl_local_for_warmup = kl_divergence_z
    kl_local_no_warmup = kl_divergence_l
    weighted_kl_local = kl_weight .* kl_local_for_warmup .+ kl_local_no_warmup

    P = compute_differentiable_transition_probs(q)
    kl_qp = tsne_repel(z, P) * min(epoch, size(q,1)) #min(epoch, m.n_hidden[end])

    lossval = mean(reconst_loss + weighted_kl_local) + kl_qp
    return lossval
end

# without differentiation through P 
function scvis_loss(m::scVAE, x::AbstractMatrix{S}, P::AbstractMatrix{S}; kl_weight::Float32=1.0f0, epoch::Int=1) where S <: Real 
    z, qz_m, qz_v, ql_m, ql_v, library = scVI.inference(m, x)
    # size(z) = latent_dim x batchsize
    px_scale, px_r, px_rate, px_dropout = scVI.generative(m, z, library)
    kl_divergence_z = -0.5f0 .* sum(1.0f0 .+ log.(qz_v) - qz_m.^2 .- qz_v, dims=1) # 2 

    if !m.use_observed_lib_size # TODO!
        local_library_log_means,local_library_log_vars = _compute_local_library_params() 
        kl_divergence_l = kl(Normal(ql_m, ql_v.sqrt()),Normal(local_library_log_means, local_library_log_vars.sqrt())).sum(dim=1)
    else
        kl_divergence_l = 0.0f0
    end

    reconst_loss = scVI.get_reconstruction_loss(m, x, px_rate, px_r, px_dropout)
    kl_local_for_warmup = kl_divergence_z
    kl_local_no_warmup = kl_divergence_l
    weighted_kl_local = kl_weight .* kl_local_for_warmup .+ kl_local_no_warmup

    kl_qp = tsne_repel(z, P) * min(epoch, size(m.z_encoder.encoder[end][1].weight,1)) #min(epoch, m.n_hidden[end])

    lossval = mean(reconst_loss + weighted_kl_local) + kl_qp
    return lossval
end