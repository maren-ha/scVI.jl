function inference(m::scVAE, x::AbstractMatrix{S}) where S <: Real 
    encoder_input = m.log_variational ? log.(one(S) .+ x) : x
    qz_m, qz_v, z = m.z_encoder(encoder_input)
    ql_m, ql_v = nothing, nothing 
    library = get_library(m, x, encoder_input)
    return z, qz_m, qz_v, ql_m, ql_v, library
end

function get_library(m::scVAE, x::AbstractMatrix{S}, encoder_input::AbstractMatrix{S}) where S <: Real
    library = get_library(Val(m.use_observed_lib_size), x, encoder_input)
    return library
end

function get_library(::Val{true}, x::AbstractMatrix{S}, encoder_input::AbstractMatrix{S}) where S <: Real
    library = log.(sum(x, dims=1))
    return library
end

function get_library(::Val{:false}, x::AbstractMatrix{S}, encoder_input::AbstractMatrix{S}) where S <: Real
    ql_m, ql_v, library_encoded = m.l_encoder(encoder_input)
    library = library_encoded
    return library
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

function loss(m::scVAE, x::AbstractMatrix{S}; kl_weight::Float32=1.0f0) where S <: Real 
    z, qz_m, qz_v, ql_m, ql_v, library = inference(m, x)
    px_scale, px_r, px_rate, px_dropout = generative(m, z, library)
    kl_divergence_z = -0.5f0 .* sum(1.0f0 .+ log.(qz_v) - qz_m.^2 .- qz_v, dims=1) # 2 
    # equivalent to kl_divergence_z = torch.distributions.kl.kl_divergence(torch.distributions.Normal(qz_m, qz_v.sqrt()), torch.distributions.Normal(mean, scale)).sum(dim=1)

    if !m.use_observed_lib_size # TODO!
        local_library_log_means,local_library_log_vars = _compute_local_library_params() 
        kl_divergence_l = kl(Normal(ql_m, ql_v.sqrt()),Normal(local_library_log_means, local_library_log_vars.sqrt())).sum(dim=1)
    else
        kl_divergence_l = 0.0f0
    end

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

function get_reconstruction_loss(::Val{:poisson}, x::AbstractMatrix{S}, px_rate::AbstractMatrix{S}, px_r::Union{AbstractArray{S}}, px_dropout::Union{AbstractMatrix{S}}) where S <: Real 
    #error("not yet implemented")
    reconst_loss = sum(-log_poisson(x, px_rate), dims=1)
    return reconst_loss
end

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
    get_latent_representation(m::scVAE, countmatrix::Matrix; cellindices=nothing, give_mean::Bool=true)
"""
function get_latent_representation(m::scVAE, countmatrix::Matrix; cellindices=nothing, give_mean::Bool=true)
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
