#using Printf: @sprintf
rescale(A; dims=1) = (A .- mean(A, dims=dims)) ./ max.(std(A, dims=dims), eps())

"""
    Compute the point perplexities `P` given its squared distances to the other points `D`
    and the precision of Gaussian distribution `beta`.


    Calculates the perplexity of the data in `D` given the probability distribution `P` and the precision parameter `beta`.
    Mutates the input `P` to be the probability distribution.
    Raises an error if the probability distribution is degenerate or if the perplexity is not a finite number.
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

    Compute the perplexity of the similarity between each pair of points in the dataset, given a symmetric and positive definite distance matrix D.
    
    Parameters:
    - D::AbstractMatrix{T} : symmetric and positive definite distance matrix
    - tol::Number : tolerance for stopping the perplexity search (default 1e-5)
    - perplexity::Number : target perplexity (default 30.0)
    - max_iter::Integer : maximum number of iterations for the perplexity search (default 50)
    - verbose::Bool : flag to print information about the progress of the search (default false)

    Returns:
    - P::Matrix{T} : matrix of perplexities 
    - beta::Vector{T} : vector of precisions for each point
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

"""
Computes the transition probabilities between datapoints in X using the perplexity based similarity approach.

Parameters:
- X: The data matrix with shape obs x vars.
- perplexity: The perplexity value to use for computing the transition probabilities. Default is 30.

Returns:
- A matrix of transition probabilities with shape obs x obs, where P[i,j] represents the transition probability from 
  datapoint i to j.
"""
function compute_transition_probs(X, perplexity::Number=30) # X: shape obs x vars
    X = X * (1.0/std(X)::eltype(X))
    D = pairwise(SqEuclidean(), X)
    P, beta = perplexities(D, perplexity)
    return Float32.(P)
end

"""
Compute the t-SNE cost function with repulsion term.

Args:
- z: matrix of shape (latent_dim, batchsize) representing the low-dimensional
    embedding of the data.
- P: matrix of shape (batchsize, batchsize) representing the symmetric 
    pairwise affinities of the data points.

Returns:
- The t-SNE cost function value normalized by the batchsize.
"""
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
"""
    Computes the loss value of the scVI model.

    It takes in an scVAE model, a matrix of input data (x), and an optional matrix of transition probabilities (P).
    It applies the encoder to the input data to obtain a set of parameters for the approximate posterior of the latent variable (z).
    It then uses these parameters to perform the reparameterization trick to sample from the approximate posterior.
    It then applies the generative model to the sampled latent variable to obtain the parameters of the likelihood.
    It then computes the KL divergence between the approximate posterior and the prior, and the reconstruction loss between the input data and the likelihood.
    It also computes the tsne_repel loss which is the repulsion loss between the latent representations of the data.
    Finally, it returns the sum of reconstruction loss, KL divergence, and tsne_repel loss normalized by the batch size.

    Parameters:
    - m: an instance of the scVAE model
    - x: the input data, a matrix of shape obs x vars
    - P: (Optional) the transition probabilities matrix, defaults to nothing
    - kl_weight: (Optional) weight for the KL divergence term, defaults to 1.0
    - epoch: (Optional) current training epoch, defaults to 1
    Returns:
    - lossval: the loss value

    Parameters:
    - m: an instance of the scVAE model
    - x: input data of shape obs x vars
    - P: (Optional) matrix of transition probabilities. If not provided, it will be computed internally.
    - kl_weight: weighting factor for KL divergence. 
    - epoch: The current epoch number.
    
    Returns:
    - lossval: The final loss value
"""
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