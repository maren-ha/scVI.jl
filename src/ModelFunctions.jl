#-------------- encoder and decoder -------------------------------

function inference(m::scVAE, x::AbstractMatrix{S}) where S <: Real 
    encoder_input = m.log_variational ? log.(one(S) .+ x) : x
    qz_m, qz_v, z = m.z_encoder(encoder_input)
    #ql_m, ql_v = nothing, nothing 
    ql_m, ql_v, library = get_library(m, x, encoder_input)
    return z, qz_m, qz_v, ql_m, ql_v, library
end

function generative(m::scVAE, z::AbstractMatrix{S}, library::AbstractMatrix{S}) where S <: Real 
    decoder_input = z
    # categorical_input=()
    px_scale, px_r, px_rate, px_dropout = m.decoder(decoder_input, library)
    #if m.dispersion == :gene # some other cases (:gene-batch, :gene-label ignored)
    #    px_r = m.px_r
    #end
    px_r = exp.(px_r)
    return px_scale, px_r, px_rate, px_dropout
end

#-------------- library size and encoding -------------------------------

function get_library(m::scVAE, x::AbstractMatrix{S}, encoder_input::AbstractMatrix{S}) where S <: Real
    return get_library(Val(m.use_observed_lib_size), m, x, encoder_input)
end

function get_library(::Val{true}, m::scVAE, x::AbstractMatrix{S}, encoder_input::AbstractMatrix{S}) where S <: Real
    library = log.(sum(x, dims=1))
    return nothing, nothing, library
end

function get_library(::Val{:false}, m::scVAE, x::AbstractMatrix{S}, encoder_input::AbstractMatrix{S}) where S <: Real
    ql_m, ql_v, library_encoded = m.l_encoder(encoder_input)
    library = library_encoded
    return ql_m, ql_v, library
end

KL(P::Distributions.Normal, Q::Distributions.Normal) = log(Q.σ / P.σ) + (1/2) * ((P.σ / Q.σ)^2 + (P.μ - Q.μ)^2 * Q.σ^(-2) -1.)
#P = Normal.([1, 2, 3], [0.2, 0.2, 0.1]); Q = Normal.([2, 1, 4], [0.1, 0.4, 0.2]); broadcast(KL, P, Q)

function get_kl_divergence_l(m::scVAE, ql_m::Union{Nothing, AbstractMatrix{S}}, ql_v::Union{Nothing, AbstractMatrix{S}}, batch_indices::Vector{Int}) where S <: Real 
    return get_kl_divergence_l(Val(m.use_observed_lib_size), m, ql_m, ql_v, batch_indices)
end

get_kl_divergence_l(::Val{true}, m::scVAE, ql_m::Union{Nothing, AbstractMatrix{S}}, ql_v::Union{Nothing, AbstractMatrix{S}}, batch_indices::Vector{Int}) where S <: Real = 0.0f0

function get_kl_divergence_l(::Val{false}, m::scVAE, ql_m::Union{Nothing, AbstractMatrix{S}}, ql_v::Union{Nothing, AbstractMatrix{S}}, batch_indices::Vector{Int}) where S <: Real
    local_library_log_means,local_library_log_vars = _compute_local_library_params(m, batch_indices) # https://github.com/scverse/scvi-tools/blob/765bf838483dc0e295fbedef2d003c5410af5a8f/scvi/module/_vae.py#L279 
    kl_divergence_l = sum(broadcast(KL, Normal.(ql_m, sqrt.(ql_v)), Normal.(local_library_log_means, sqrt.(local_library_log_vars))))
    return kl_divergence_l
end

function _compute_local_library_params(m::scVAE, batch_indices::Vector{S}) where S <: Real 
        """Computes local library parameters.
        Compute two vectors of length = length(batch_indices) where each
        element corresponds to the mean and variances, respectively, of the
        log library sizes in the batch the cell corresponds to.
        """
        n_batch = m.n_batch
        local_library_log_means = onehotbatch(batch_indices, collect(1:n_batch))' * m.library_log_means
        # onehotbatch: size n_batch x batchsize, library_log_means: length n_batch 
        local_library_log_vars = onehotbatch(batch_indices, collect(1:n_batch))' * m.library_log_vars
        return local_library_log_means, local_library_log_vars # length: batchsize 
end

#-------------- reconstruction loss -------------------------------

function get_reconstruction_loss(m::scVAE, x::AbstractMatrix{S}, px_rate::AbstractMatrix{S}, px_r::Union{AbstractArray{S}, Nothing}, px_dropout::Union{AbstractMatrix{S}, Nothing}) where S <: Real 
    return get_reconstruction_loss(Val(m.gene_likelihood), x, px_rate, px_r, px_dropout)
end

function get_reconstruction_loss(::Val{:zinb}, x::AbstractMatrix{S}, px_rate::AbstractMatrix{S}, px_r::AbstractArray{S}, px_dropout::AbstractMatrix{S}) where S <: Real 
    reconst_loss = sum(-log_zinb_positive(x, px_rate, px_r, px_dropout), dims=1)
    return reconst_loss
end

function get_reconstruction_loss(::Val{:nb}, x::AbstractMatrix{S}, px_rate::AbstractMatrix{S}, px_r::AbstractArray{S}, px_dropout::Union{AbstractMatrix{S}, Nothing}) where S <: Real 
    reconst_loss = sum(-log_nb_positive(x, px_rate, px_r), dims=1)
    return reconst_loss
end

function get_reconstruction_loss(::Val{:poisson}, x::AbstractMatrix{S}, px_rate::AbstractMatrix{S}, px_r::Union{AbstractArray{S}, Nothing}, px_dropout::Union{AbstractMatrix{S}, Nothing}) where S <: Real 
    reconst_loss = sum(-log_poisson(x, px_rate), dims=1)
    return reconst_loss
end

function get_reconstruction_loss(::Val{:gaussian}, x::AbstractMatrix{S}, px_rate::AbstractMatrix{S}, px_r::AbstractArray{S}, px_dropout::Union{AbstractMatrix{S}, Nothing}) where S <: Real 
    reconst_loss = sum(-log_normal(x, px_rate, px_r), dims=1)
    return reconst_loss
end

function get_reconstruction_loss(::Val{:bernoulli}, x::AbstractMatrix{S}, px_rate::AbstractMatrix{S}, px_r::Union{AbstractArray{S}, Nothing}, px_dropout::Union{AbstractMatrix{S}, Nothing}) where S <: Real 
    reconst_loss = sum(-log_binary(x, px_rate), dims=1)
    return reconst_loss
end

#-------------- standard overall loss  -------------------------------

loss(m::scVAE, d::Tuple; kl_weight::Float32=1.0f0) = loss(m, d...; kl_weight=kl_weight)

loss(m::scVAE, x::AbstractMatrix{S}; kl_weight::Float32=1.0f0) where S <: Real = loss(m, x, fill(1, size(x,2)); kl_weight=kl_weight)

function loss(m::scVAE, x::AbstractMatrix{S}, batch_indices::Vector{Int}; kl_weight::Float32=1.0f0) where S <: Real 
    z, qz_m, qz_v, ql_m, ql_v, library = inference(m, x)
    px_scale, px_r, px_rate, px_dropout = generative(m, z, library)
    kl_divergence_z = -0.5f0 .* sum(1.0f0 .+ log.(qz_v) - qz_m.^2 .- qz_v, dims=1) # 2 
    # equivalent to kl_divergence_z = torch.distributions.kl.kl_divergence(torch.distributions.Normal(qz_m, qz_v.sqrt()), torch.distributions.Normal(mean, scale)).sum(dim=1)

    kl_divergence_l = get_kl_divergence_l(m, ql_m, ql_v, batch_indices)

    reconst_loss = get_reconstruction_loss(m, x, px_rate, px_r, px_dropout)
    kl_local_for_warmup = kl_divergence_z
    kl_local_no_warmup = kl_divergence_l
    weighted_kl_local = kl_weight .* kl_local_for_warmup .+ kl_local_no_warmup

    lossval = mean(reconst_loss + weighted_kl_local)
    #kl_local = Dict("kl_divergence_l" => kl_divergence_l, "kl_divergence_z" => kl_divergence_z)
    #kl_global = [0.0]
    #return lossval#, reconst_loss, kl_local, kl_global
    return lossval
end

#-------------- loss registry -------------------------------

register_losses!(m::scVAE, d::Tuple; kl_weight::Float32=1.0f0) = register_losses!(m, d...; kl_weight=kl_weight)

register_losses!(m::scVAE, x::AbstractMatrix{S}; kl_weight::Float32=1.0f0) where S <: Real = register_losses!(m, x, fill(1, size(x,2)); kl_weight=kl_weight)

function register_losses!(m::scVAE, x::AbstractMatrix{S}, batch_indices::Vector{Int}; kl_weight::Float32=1.0f0) where S <: Real 
    z, qz_m, qz_v, ql_m, ql_v, library = inference(m, x)
    px_scale, px_r, px_rate, px_dropout = generative(m, z, library)
    kl_divergence_z = -0.5f0 .* sum(1.0f0 .+ log.(qz_v) - qz_m.^2 .- qz_v, dims=1) # 2 

    kl_divergence_l = get_kl_divergence_l(m, ql_m, ql_v, batch_indices)

    reconst_loss = get_reconstruction_loss(m, x, px_rate, px_r, px_dropout)
    kl_local_for_warmup = kl_divergence_z
    kl_local_no_warmup = kl_divergence_l
    weighted_kl_local = kl_weight .* kl_local_for_warmup .+ kl_local_no_warmup

    lossval = mean(reconst_loss + weighted_kl_local)

    push!(m.loss_registry["kl_z"], mean(kl_divergence_z))
    push!(m.loss_registry["kl_l"], mean(kl_divergence_l))
    push!(m.loss_registry["reconstruction"], mean(reconst_loss))
    push!(m.loss_registry["total_loss"], lossval)
    return m
end

#-------------- supervised loss -------------------------------

supervised_loss(m::scVAE, x::AbstractMatrix{S}, y::AbstractMatrix{S}; kl_weight::Float32=1.0f0) where S <: Real = supervised_loss(m, x, y, fill(1, size(x,2)); kl_weight=kl_weight)

function supervised_loss(m::scVAE, x::AbstractMatrix{S}, y::AbstractMatrix{S}, batch_indices::Vector{Int}; kl_weight::Float32=1.0f0) where S <: Real 
    z, qz_m, qz_v, ql_m, ql_v, library = inference(m,x)
    px_scale, px_r, px_rate, px_dropout = generative(m, z, library)
    kl_divergence_z = -0.5f0 .* sum(1.0f0 .+ log.(qz_v) - qz_m.^2 .- qz_v, dims=1) 

    kl_divergence_l = get_kl_divergence_l(m, ql_m, ql_v, batch_indices)

    reconst_loss = get_reconstruction_loss(m, x, px_rate, px_r, px_dropout)
    kl_local_for_warmup = kl_divergence_z
    kl_local_no_warmup = kl_divergence_l
    weighted_kl_local = kl_weight .* kl_local_for_warmup .+ kl_local_no_warmup

    enc_loss = Flux.mse(qz_m, y)
    lossval = mean(reconst_loss + weighted_kl_local)
    total_loss = enc_loss + lossval
    return total_loss
end

#-------------- misc: kl_weight and latent representation -------------------------------

function get_kl_weight(n_epochs_kl_warmup, n_steps_kl_warmup, current_epoch, global_step)
    epoch_criterion =!isnothing(n_epochs_kl_warmup)
    step_criterion = !isnothing(n_steps_kl_warmup)
    if epoch_criterion
        kl_weight = min(1.0f0, Float32(current_epoch / n_epochs_kl_warmup))
    elseif step_criterion
        kl_weight = min(1.0f0, Float32(global_step / n_steps_kl_warmup))
    else
        kl_weight = 1.0f0
    end 
    return kl_weight 
end

"""
    get_latent_representation(m::scVAE, countmatrix::Matrix; 
        cellindices=nothing, give_mean::Bool=true
    )

Computes the latent representation of an `scVAE` model on input count data by applying the `scVAE` encoder. 

Returns the mean (default) or a sample of the latent representation (can be controlled by `give_mean` keyword argument).

**Arguments:**
-----------------
 - `m::scVAE`: `scVAE` model from which the encoder is applied to get the latent representation
 - `countmatrix::Matrix`: matrix of counts (e.g., `countmatrix` field of an `AnnData` object), which is to be embedded with the `scVAE` model encoder. Is assumed to be in a (cell x gene) format.

 **Keyword arguments:**
 -----------------
 - `cellindices=nothing`: optional; indices of cells (=rows) on which to subset the `countmatrix` before embedding it 
 - `give_mean::Bool=true`: optional; if `true`, returns the mean of the latent representation, else returns a sample. 
"""
function get_latent_representation(m::scVAE, countmatrix::AbstractArray; 
    cellindices=nothing, give_mean::Bool=true
    )
    # countmatrix assumes cells x genes 
    if !isnothing(cellindices)
        countmatrix = countmatrix[cellindices,:]
    end
    z, qz_m, qz_v, ql_m, ql_v, library = inference(m,countmatrix')
    if give_mean
        return qz_m
    else
        return z
    end
end
