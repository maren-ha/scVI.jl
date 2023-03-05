function Hbeta_diff!(P::AbstractVector, D::AbstractVector, beta::Number)
    #@inbounds P .= exp.(-beta .* D) # exp(|x_i - x_j|^2 * 1/σ^2) 5000 x 5000 
    P = exp.(-beta .* D)
    sumP = sum(P)
    H = log(sumP) + beta * dot(D, P) / sumP  # n_samples, H(x) = ΣP(x)log(1/P(x)), here Pj|i = exp(|x_i - x_j|^2 * 1/σ^2)/Σexp(|x_i - x_j|^2 * 1/σ^2)
    #@inbounds P .*= 1/sumP # n x n
    P = map(x -> x * (1/sumP), P)
    return H, P
end

function do_perplexity_loop(i, D::AbstractMatrix{S}, Pcol::AbstractArray{S}, Htarget, tol, max_iter) where S <: Real 
    # Compute the Gaussian kernel and entropy for the current precision (1/σ)
    betai = one(S)
    betamin = zero(S)
    betamax = S.(Inf)

    #Di = map(ind -> ind == i ? prevfloat(Inf) : Di[ind], collect(1:length(Di))) # exclude D[i,i] from minimum(), yet make it finite and exp(-D[i,i])==0.0
    Di = map(ind -> ind == i ? S.(9e20) : D[ind,i], collect(1:size(D,1))) # exclude D[i,i] from minimum(), yet make it finite and exp(-D[i,i])==0.0
    minD = minimum(Di) # distance of i-th point to its closest neighbour
    Di = map(x -> x - minD, Di) # @inbounds Di .-= minD # cells entropy is invariant to offsetting Di, which helps to avoid overflow

    H, Pcol = Hbeta_diff!(Pcol, Di, betai)
    Hdiff = H - Htarget # n_samples

    # Evaluate whether the perplexity is within tolerance
    tries = 0
    while abs.(Hdiff) > tol && tries < max_iter
        # If not, increase or decrease precision
        if Hdiff > zero(S)
            betamin = betai
            betai = isfinite(betamax) ? (betai + betamax)/2 : betai*2
        else
            betamax = betai
            betai = (betai + betamin)/2
        end

        # Recompute the values
        H, Pcol = Hbeta_diff!(Pcol, Di, betai)
        Hdiff = H - Htarget
        tries += 1
    end
    return Pcol, betai
end

"""
    Compute the differentiable perplexities of a symmetric and positive distance matrix D.
    This function compute the differentiable perplexities for the given symmetric and positive distance matrix D. 
        The perplexity is a scalar value that describes the effective number of nearest neighbours for each point in the data set. 
        The perplexity can be used to control the balance between preserving the local structure of the data and the global structure of the data.
    
    Parameters:
    - D : AbstractMatrix{S} : symmetric and positive distance matrix 
    - tol : S : tolerance for the perplexity error (default: 1e-5)
    - perplexity : S : target perplexity (default: 30.0)
    - max_iter : Integer : maximum number of iterations (default: 50)
    - verbose : Bool : print the progress of the calculation (default: false)
    
    Returns:
    - P : matrix :  final P-matrix
    - beta : vector : the final beta value for each datapoint
"""
function differentiable_perplexities(D::AbstractMatrix{S}, tol::S = Float32(1e-5), perplexity::S = 30.0f0;
    max_iter::Integer = 50,
    verbose::Bool=false) where S<:Real
    if !(issymmetric(D) && all(x -> x >= zero(S), D))
        error("Distance matrix D must be symmetric and positive")
    end
    n = size(D, 1)
    # initialize
    Pcol = fill(zero(S), n)
    Htarget = log(perplexity) # perplexity = 2^H(P)
    # Loop over all datapoints
    #results = hcat(collect.([do_perplexity_loop(i, D, Pcol, Htarget, tol, max_iter) for i in 1:n])...)
    results = hcat([do_perplexity_loop(i, D, Pcol, Htarget, tol, max_iter) for i in 1:n]...)[1,:]
    # Return final P-matrix
    P = hcat(collect(results[i][1] for i in 1:length(results))...)
    beta = collect(results[i][2] for i in 1:length(results))

    return P, beta
end

"""
    Compute differentiable transition probabilities from a given matrix X.
    Parameters:
    -----------
    X : AbstractMatrix{S}
        The matrix from which to compute the differentiable transition probabilities.
    perplexity : S
        The perplexity to use for the computation of the transition probabilities.
    Returns:
    --------
    P : matrix of type S
        The differentiable transition probabilities.
"""
function compute_differentiable_transition_probs(X::AbstractMatrix{S}, perplexity::S=30.0f0) where S <: Real # X: shape obs x vars
    X = X * (one(S)/std(X)::eltype(X))
    D = pairwise(SqEuclidean(), X)
    P, beta = differentiable_perplexities(D, S.(1e-5), perplexity)
    return Float32.(P)
end

"""
Computes the t-SNE loss for the given scVAE model, tsne network, input data, and transition probabilities.

Parameters:
    - m: scVAE model
    - tsne_net: Dense network for t-SNE
    - x: Input data, shape obs x vars 
    - P: Transition probabilities, shape obs x obs, defaults to `Nothing`
    - epoch: Current training epoch
    
Returns:
    - kl_qp: t-SNE loss
"""
function tsne_loss(m::scVAE, tsne_net::Dense, x::AbstractMatrix{S}, P::Union{Nothing, AbstractMatrix{S}}, epoch::Int=1) where S <: Real 
    z, qz_m, qz_v, ql_m, ql_v, library = scVI.inference(m, x)
    latent = z # alternative: qz_m? 
    z_tsne = tsne_net(latent)
    if isnothing(P)
        P = compute_differentiable_transition_probs(latent)
    end
    kl_qp = tsne_repel(z_tsne, P) * S.(min(epoch, m.n_latent)) #min(epoch, m.n_hidden[end])
end

#function tsne_loss(m::scVAE, tsne_net::Dense, x::AbstractMatrix{S}, epoch::Int=1) where S <: Real 
#    z, qz_m, qz_v, ql_m, ql_v, library = scVI.inference(m, x)
#    latent = z # alternative: qz_m? 
#    z_tsne = tsne_net(latent)
#    P = compute_differentiable_transition_probs(latent')
#    kl_qp = tsne_repel(z_tsne, P) * S.(min(epoch, m.n_latent)) #min(epoch, m.n_hidden[end])
#end