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
                      verbose::Bool=false, progress::Bool=true) where T<:Number
    (issymmetric(D) && all(x -> x >= 0, D)) ||
        throw(ArgumentError("Distance matrix D must be symmetric and positive"))

    # initialize
    n = size(D, 1)
    P = fill(zero(T), n, n) # perplexities matrix
    beta = fill(one(T), n)  # vector of Normal distribution precisions for each point
    Htarget = log(perplexity) # the expected entropy
    Di = fill(zero(T), n)
    Pcol = fill(zero(T), n)

    # Loop over all datapoints
    progress && (pb = Progress(n, "Computing point perplexities"))
    for i in 1:n
        progress && ProgressMeter.update!(pb, i)

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
        verbose && abs(Hdiff) > tol && @warn "P[$i]: perplexity error is above tolerance: $(Hdiff)"
        # Set the final column of P
        @assert Pcol[i] == 0.0 "Diagonal probability P[$i,$i]=$(Pcol[i]) not zero"
        @inbounds P[:, i] .= Pcol
        beta[i] = betai
    end
    progress && finish!(pb)
    # Return final P-matrix
    verbose && @info("Mean σ=$(mean(sqrt.(1 ./ beta)))")
    return P, beta
end

function kldivel(p, q)
    if (p > zero(p) && q > zero(q))
        return p*log(p/q)
    else 
        return zero(p)
    end
end

function compute_kldiv(z, P, sum_P)
    Q = z'*z
    sum_Q = 0.0
    kldiv = 0.0
    for j in 1:size(Q, 2)
        Pj = view(P, :, j)
        sum_YYj_p1 = 1.0 + Q[j, j]
        Qj = view(Q, :, j)
        Qjj = 0.0
        kldiv_j = 0.0
        for i in 1:(j-1)
            sqdist_p1 = sum_YYj_p1 - 2.0 * Qj[i] + Q[i, i]
            @fastmath Qji = ifelse(sqdist_p1 > 1.0, 1.0 / sqdist_p1, 1.0)
            sum_Q += Qji
            @fastmath kldiv_j += kldivel(Pj[i], Qji)
        end
        kldiv += 2*kldiv_j + kldivel(Pj[j], Q[j])
    end
    sum_Q *= 2 # the diagonal and lower-tri part of Q is zero
    last_kldiv = kldiv/sum_P + log(sum_Q/sum_P) # adjust wrt P and Q scales
    return last_kldiv
end

function loss(m::scVAE, x::AbstractMatrix{S}, P::AbstractMatrix{S}, batch_indices::Vector{Int}; kl_weight::Float32=1.0f0, cheat::Bool=true, cheat_scale::Float32 = 12.0f0) where S <: Real 
    z, qz_m, qz_v, ql_m, ql_v, library = scVI.inference(m, x)
    px_scale, px_r, px_rate, px_dropout = scVI.generative(m, z, library)
    kl_divergence_z = -0.5f0 .* sum(1.0f0 .+ log.(qz_v) - qz_m.^2 .- qz_v, dims=1) # 2 

    kl_divergence_l = scVI.get_kl_divergence_l(m, ql_m, ql_v, batch_indices)

    reconst_loss = scVI.get_reconstruction_loss(m, x, px_rate, px_r, px_dropout)
    kl_local_for_warmup = kl_divergence_z
    kl_local_no_warmup = kl_divergence_l
    weighted_kl_local = kl_weight .* kl_local_for_warmup .+ kl_local_no_warmup

    P = P[1:size(P,2),:] .+ P[1:size(P,2),:]'
    if cheat
        P = P .* cheat_scale/sum(P) # normalize + early exaggeration
        sum_P = cheat_scale
    end
    tsne_penalty = scVI.compute_kldiv(z, P, sum_P)
    println(tsne_penalty)

    #graph_loss = sum(z*(Diagonal(vec(sum(P, dims=1))) .- P)*z')
    #0.5.*sum(P[i,j].*(z[:,i] .- z[:,j]).^2 for i in 1:size(P,1), j in 1:size(P,2))
    #println(graph_loss)

    lossval = mean(reconst_loss + weighted_kl_local) 
    return lossval + 150.0f0*tsne_penalty #+ 1.0f0 * graph_loss
end

function register_losses!(m::scVAE, x::AbstractMatrix{S}, P::AbstractMatrix{S}, batch_indices::Vector{Int}; 
    kl_weight::Float32=1.0f0, cheat::Bool=true, cheat_scale::Float32=12.0f0) where S <: Real 
    z, qz_m, qz_v, ql_m, ql_v, library = inference(m, x)
    px_scale, px_r, px_rate, px_dropout = generative(m, z, library)
    kl_divergence_z = -0.5f0 .* sum(1.0f0 .+ log.(qz_v) - qz_m.^2 .- qz_v, dims=1) # 2 

    kl_divergence_l = get_kl_divergence_l(m, ql_m, ql_v, batch_indices)

    reconst_loss = get_reconstruction_loss(m, x, px_rate, px_r, px_dropout)
    kl_local_for_warmup = kl_divergence_z
    kl_local_no_warmup = kl_divergence_l
    weighted_kl_local = kl_weight .* kl_local_for_warmup .+ kl_local_no_warmup

    P = P[1:size(P,2),:] .+ P[1:size(P,2),:]'
    if cheat
        P = P .* cheat_scale/sum(P) # normalize + early exaggeration
        sum_P = cheat_scale
    end
    tsne_penalty = compute_kldiv(z, P, sum_P)

    lossval = mean(reconst_loss + weighted_kl_local) + 100.0f0*tsne_penalty

    push!(m.loss_registry["kl_z"], mean(kl_divergence_z))
    push!(m.loss_registry["kl_l"], mean(kl_divergence_l))
    push!(m.loss_registry["reconstruction"], mean(reconst_loss))
    push!(m.loss_registry["tSNE_loss"], tsne_penalty)
    push!(m.loss_registry["total_loss"], lossval)
    return m
end

function train_tSNE_model!(m::scVAE, adata::AnnData, training_args::TrainingArgs; 
    layer::Union{String, Nothing}=nothing, 
    batch_key::Symbol=:batch,
    perplexity::Number=30.0, 
    cheat_scale::Number=12.0, 
    cheat::Bool=true)

    opt = Flux.Optimiser(Flux.Optimise.WeightDecay(training_args.weight_decay), ADAM(training_args.lr))
    ps = Flux.params(m)

    # get tSNE similarity matrix 
    if !haskey(adata.layers, "rescaled")
        rescale!(adata)
    end
    X_tSNE = adata.layers["rescaled"]

    # prepare P matrix of high-dimensional similarities
    X_tSNE = X_tSNE * (1.0/std(X_tSNE)::eltype(X_tSNE)) # note that X is copied
    D = pairwise(SqEuclidean(), X_tSNE')
    P, beta = perplexities(D, 1e-5, perplexity, verbose=training_args.verbose, progress=training_args.progress)
    P .+= P' # make P symmetric
    P = Float32.(P)
    sum_P = sum(P)

    # get matrix on which to operate 
    if m.gene_likelihood ∈ [:gaussian, :bernoulli]
        isnothing(layer) && throw(ArgumentError("If using Gaussian or Bernoulli generative distribution, the adata layer on which to train has to be specified explicitly"))
        X = adata.layers[layer]
    else
        X = adata.X
    end

    ncells, ngenes = size(X)

    if training_args.train_test_split
        trainsize = training_args.trainsize
        validationsize = 1 - trainsize # nothing 
        train_inds = shuffle!(collect(1:ncells)[1:Int(ceil(trainsize*ncells))])
    else
        train_inds = collect(1:ncells);
    end

    if training_args.register_losses
        m.loss_registry["kl_l"] = []
        m.loss_registry["kl_z"] = []
        m.loss_registry["reconstruction"] = []
        m.loss_registry["total_loss"] = []
        m.loss_registry["tSNE_loss"] = []
    end

    batch_indices = setup_batch_indices_for_library_scaling(m, adata, batch_key, verbose=training_args.verbose)

    train_steps=0
    @info "Starting training for $(training_args.max_epochs) epochs..."
    progress = training_args.progress && Progress(length(Iterators.partition(train_inds, training_args.batchsize))*training_args.max_epochs);

    for epoch in 1:training_args.max_epochs
        if epoch > 50
            cheat = false
        end
        kl_weight = get_kl_weight(training_args.n_epochs_kl_warmup, training_args.n_steps_kl_warmup, epoch, 0)
        for inds in Iterators.partition(shuffle(train_inds), training_args.batchsize)
            d = X[inds,:]'
            batch_inds = batch_indices[inds]
            Pmat = P[inds, inds]
            curloss, back = Flux.pullback(ps) do 
                loss(m, d, Pmat, batch_inds; kl_weight=kl_weight, cheat=cheat)
            end
            grad = back(1f0)
            Flux.Optimise.update!(opt, ps, grad)
            if training_args.progress
                next!(progress; showvalues=[(:loss, curloss)]) 
            elseif (train_steps % training_args.verbose_freq == 0) && training_args.verbose
                @info "epoch $(epoch):" loss=curloss
            end
            train_steps += 1
        end
        training_args.register_losses && register_losses!(m, Float32.(X[train_inds,:]'), P[train_inds, train_inds], batch_indices[train_inds]; kl_weight=kl_weight)
    end
    @info "training complete!"
    m.is_trained = true
    return m, adata
end