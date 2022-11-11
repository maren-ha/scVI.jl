#-------------------------------------------------------------------------------------
# Utils 
#-------------------------------------------------------------------------------------
# mu + eps * exp(0.5 * var)
function reparameterize_gaussian(mean, var)
    # Julia Distributions, like torch, parameterizes the Normal with std, not variance
    # Normal(μ, σ)      # Normal distribution with mean μ and variance σ^2
    return mean + sqrt.(var) .* randn(Float32, size(mean))
    #Normal(mu, var.sqrt()).rsample() # = mu + var.sqrt() * eps where eps = standard_normal(shape_of_sample)  
end
#TODO write String Doc
"""
Compute the point perplexities `P` given its squared distances to the other points `D`
and the precision of Gaussian distribution `beta = (1/σ^2)`.

Why inverse of covariance not covariance?

Covariance matrix can represent relations between all variables 
while inverse covariance shows the relations of elements with their neighbors.
compute_entropy
"""
function Hbeta!(P::AbstractVector, D::AbstractVector, beta::Number)
    @inbounds P .= exp.(-beta .* D) # exp(|x_i - x_j|^2 * 1/σ^2) 5000 x 5000 
    sumP = sum(P)
    @assert (isfinite(sumP) && sumP > 0.0) "Degenerated P: sum=$sumP, beta=$beta"
    H = log(sumP) + beta * dot(D, P) / sumP  # n_samples, H(x) = ΣP(x)log(1/P(x)), here Pj|i = exp(|x_i - x_j|^2 * 1/σ^2)/Σexp(|x_i - x_j|^2 * 1/σ^2)
    #@assert isfinite.(H) "Degenerated H"
    @inbounds P .*= 1/sumP # n x n
    return H
end
#########
function differentiable_perplexities(D::AbstractMatrix{T}, tol::Number = 1e-5, perplexity::Number = 30.0;
    max_iter::Integer = 50,
    verbose::Bool=false) where T<:Number
    (issymmetric(D) && all(x -> x >= 0, D)) ||
    throw(ArgumentError("Distance matrix D must be symmetric and positive"))

    # initialize
    n = size(D, 1) # cells
    P = fill(zero(T), n, n) # cells x cells perplexities matrix
    beta = fill(one(T), n)  # cells vector of Normal distribution precisions (1/σ2) for each point
    Htarget = log(perplexity) # perplexity = 2^H(P)
    #Di = fill(zero(T), n) # cells 
    Pcol = fill(zero(T), n)

    # Loop over all datapoints
    # copyto!(Di, view(D, :, i)) # cells 
    # Pcol, betai = do_perplexity_loop
    # Float32.(Pcol) to fix InexactError
    results = hcat(collect.([do_perplexity_loop(i, view(D, :, i), Float32.(Pcol),Htarget, tol, max_iter, verbose) for i in 1:n])...)
    #beta = results[:,1]
    #Pcols = results[:,2]
    #P = reshape(Pcols,...)
    #@inbounds P[:, i] .= Pcol
    #beta[i] = betai
    # Return final P-matrix
    P = hcat(results[1,:]...)
    beta = results[2,:]
    # @info leads to Compiling Tuple error it must not be here 
    #TODO we might need to think about @nograd thingy
    #verbose && @info(@sprintf("Mean σ=%.4f", mean(sqrt.(1 ./ beta))))
    return P, beta
end
function do_perplexity_loop(i, Di, Pcol, Htarget, tol, max_iter,verbose)
    # Compute the Gaussian kernel and entropy for the current precision (1/σ)
    betai = 1.0
    betamin = 0.0
    betamax = Inf

    Di = map(ind -> ind == i ? prevfloat(Inf) : Di[ind], collect(1:length(Di))) # exclude D[i,i] from minimum(), yet make it finite and exp(-D[i,i])==0.0
    minD = minimum(Di) # distance of i-th point to its closest neighbour
    @inbounds Di .-= minD # cells entropy is invariant to offsetting Di, which helps to avoid overflow

    H = Hbeta!(Pcol, Di, betai)
    Hdiff = H - Htarget # n_samples

    # Evaluate whether the perplexity is within tolerance
    tries = 0
    while abs.(Hdiff) > tol && tries < max_iter
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
    verbose && abs.(Hdiff) > tol && warn("P[$i]: perplexity error is above tolerance: $(Hdiff)")
    # Set the final column of P
    #@assert Pcol[i] == 0.0 "Diagonal probability P[$i,$i]=$(Pcol[i]) not zero"

    return Pcol, betai
end

@nograd begin
    function compute_transition_probs(D) # X: shape obs x vars
        P, beta = differentiable_perplexities(D)
        return Float32.(P)
    end
end

function FCLayers(
    n_in, n_out; 
    activation_fn::Function=relu,
    bias::Bool=true,
    dropout_rate::Float32=0.1f0, 
    n_hidden::Int=128, 
    n_layers::Int=1, 
    use_batch_norm::Bool=true,
    use_layer_norm::Bool=false,
    use_activation::Bool=true,
    )

    #if n_layers != 1 
    #    @warn "n_layers > 1 currently not supported; model initialization will default to one hidden layer only"
    #end

    activation_fn = use_activation ? activation_fn : identity

    batchnorm = use_batch_norm ? BatchNorm(n_out, momentum = Float32(0.01), ϵ = Float32(0.001)) : identity
    layernorm = use_layer_norm ? LayerNorm(n_out, affine=false) : identity

    innerdims = [n_hidden for _ in 1:n_layers-1]
    layerdims = [n_in, innerdims..., n_out]
    
    layerlist = [
        Chain(
            Dense(layerdims[i], layerdims[i+1], bias=bias),
            batchnorm, 
            layernorm,
            x -> activation_fn.(x),
            Dropout(dropout_rate) # if dropout_rate > 0 
        ) 
        for i in 1:n_layers
    ]

    fc_layers = n_layers == 1 ? layerlist[1] : Chain(layerlist...)

    return fc_layers
end

# plot & save figuures 
function plot_losses(nepoch,
    moes_loss,
    loss_rnas,
    loss_proteins, figure_path)
    figure = plot(collect(1:nepoch), 
    hcat(log.(10, moes_loss .+ 1), 
    log.(10, loss_rnas .+1 ), 
    log.(10, loss_proteins .+1 )), 
    title = "Loss", 
    label=["MoE-Loss" "GEX-Loss" "Protein-Loss"], 
    xlabel="number of epoch", ylabel="log10(MultiVAE loss)", legend=:topright)
    savefig(figure,"$(figure_path)/traing_loss.png")
end 

num_params(model) = sum(length, Flux.params(model))
round4(x) = round(x, digits=4)