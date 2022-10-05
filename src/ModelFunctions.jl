function do_inference(m, x::AbstractVector{Matrix{S}}) where S <: Real
    inference(typeof(m),m,x::AbstractVector{Matrix{S}})
end

function do_generative(m, z::AbstractVector{Matrix{S}}, library::AbstractMatrix{S})where S <: Real
    generative(typeof(m),m,z,library)
end

function inference(::Type{scVAE},m, x::AbstractVector{Matrix{S}}) where S <: Real 
    # unpack the data 
    x = x[1]
    encoder_input = m.log_variational ? log.(one(S) .+ x) : x
    qz_m, qz_v, z = m.z_encoder(encoder_input)
    ql_m, ql_v = nothing, nothing 
    library = get_library(m, x, encoder_input)
    return z, qz_m, qz_v, ql_m, ql_v, library
end

function get_library(m, x::AbstractMatrix{S}, encoder_input::AbstractMatrix{S}) where S <: Real
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

function generative(::Type{scVAE},m, z::AbstractVector{Matrix{S}}, library::AbstractMatrix{S}) where S <: Real 
    decoder_input = z[1]
    # categorical_input=()
    px_scale, px_r, px_rate, px_dropout = m.decoder(decoder_input, library)
    #px_scale, px_r, px_rate = m.decoder(decoder_input, library)
    #if m.dispersion == :gene # some other cases (:gene-batch, :gene-label ignored)
    #    px_r = m.px_r
    #end
    px_r = exp.(px_r)
    return px_scale, px_r, px_rate, px_dropout
   
end
function loss(m::scVAE, x::AbstractMatrix{S}; kl_weight::Float32=1.0f0) where S <: Real
    # pack the data in an array 
    x = [x]
    z, qz_m, qz_v, ql_m, ql_v, library = do_inference(m, x)
    z = [z]
    px_scale, px_r, px_rate, px_dropout = do_generative(m, z, library)
    kl_divergence_z = -0.5f0 .* sum(1.0f0 .+ log.(qz_v) - qz_m.^2 .- qz_v, dims=1) # 2 
    # equivalent to kl_divergence_z = torch.distributions.kl.kl_divergence(torch.distributions.Normal(qz_m, qz_v.sqrt()), torch.distributions.Normal(mean, scale)).sum(dim=1)

    if !m.use_observed_lib_size # TODO!
        local_library_log_means,local_library_log_vars = _compute_local_library_params() 
        kl_divergence_l = kl(Normal(ql_m, ql_v.sqrt()),Normal(local_library_log_means, local_library_log_vars.sqrt())).sum(dim=1)
    else
        kl_divergence_l = 0.0f0
    end
    d = x[1] 
    reconst_loss = get_reconstruction_loss(m, Float32.(d), Float32.(px_rate), (px_r), (px_dropout))
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
    return get_reconstruction_loss(Val(m.modality_likelihood), x, px_rate, px_r, px_dropout)
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

function get_latent_representation(m::scVAE, countmatrix::Matrix; cellindices=nothing, give_mean::Bool=true)
    # countmatrix assumes cells x genes 
    if !isnothing(cellindices)
        countmatrix = countmatrix[cellindices,:]
    end
    x = [countmatrix'] 
    z, qz_m, qz_v, ql_m, ql_v, library = do_inference(m,x)
    if give_mean
        return qz_m
    else
        return z
    end
end
###################################################### Multimodality Functions ##########################################################################
# TODO revise the implemetation and further improve the function by using dispatching
# MoE Loss Function, calculating the recons can be better, leaving it for now ! calling nb and zinb in the function!
function loss(m::scMultiVAE_, x::AbstractVector{Matrix{S}}; kl_weight::Float32=1.0f0) where S <: Real 
    # unpack the data 
    mod1, mod2 = x[1] , x[2] 
    #apply inference on multimodality
    z_rna ,z_protein, qz_m_rna, qz_v_rna,qz_m_p, qz_v_p, ql_m, ql_v, library  = do_inference(m, x)
    z = [z_rna, z_protein]
    px_scale, px_r, px_rate, px_dropout, px_scale_, px_r_, px_rate_, px_dropout_  = do_generative(m, z,library)
    # no off diagonal calculation
    # KL divergence for GEX 
    kl_divergence_rna = -0.5f0 .* sum(1.0f0 .+ log.(qz_v_rna) - qz_m_rna.^2 .- qz_v_rna, dims=1) # batchxlatent_dimentionality
    # KL divergence for Protein
    kl_divergence_p = -0.5f0 .* sum(1.0f0 .+ log.(qz_v_p) - qz_m_p.^2 .- qz_v_p, dims=1)
    # compute reconstruction error for rna modality 
    # px_z
    #reconstruction_loss_rna = get_reconstruction_loss(m, Float32.(mod1), Float32.(px_rate), Float32.(px_r), Float32.(px_dropout), "rna")
    reconstruction_loss_rna =  sum(-log_zinb_positive(Float32.(mod1), Float32.(px_rate), Float32.(px_r), Float32.(px_dropout)), dims=1)
    # compute reconstruction error for protein modality 
    # px_z
    #reconstruction_loss_p = get_reconstruction_loss(m,Float32.(mod2),Float32.(r), Float32.(p),Float32.(px_dropout), "protein" )
    reconstruction_loss_p = sum(-log_nb_positive(Float32.(mod2), Float32.(px_rate_), Float32.(px_r_)), dims=1)
    
    # reconstraction loss, scmm coding style
    #recon_loss = sum(cat(reconstruction_loss_p,reconstruction_loss_rna; dims=1),dims=1) 
    #kl_loss = sum(cat(kl_divergence_rna,kl_divergence_p; dims=1),dims=1)
    
    recon_loss = sum(reconstruction_loss_p,dims=1) + sum(reconstruction_loss_rna,dims=1) # equivalent to scmm but less code 
    kl_loss = sum(kl_divergence_rna,dims=1) + sum(kl_divergence_p,dims=1)
    multimodal_total_loss = sum(0.5 * (recon_loss + (kl_weight .* kl_loss)))
    return mean(multimodal_total_loss) 
end

#### multimodal inferenc ####
function inference(::Type{scMultiVAE_},m, x::AbstractVector{Matrix{S}}) where S <: Real
    
    mod1, mod2 = x[1], x[2]

    encoder_input = m.log_variational ? log.(one(S) .+ mod1) : mod1
    # unpack the encoders ... 
    z_encoder_g, z_encoder_p = m.z_encoder[1], m.z_encoder[2]
    
    # Modality 1  gex
    qz_m, qz_v, z_gex = z_encoder_g(encoder_input)
    # no library size estimation needed
    ql_m, ql_v = nothing, nothing 
    library = get_library(m, mod1, encoder_input)
    
    # Modality 2  protein
    encoder_input_2 = m.log_variational ? log.(one(S) .+ mod2) : mod2
    qz_m_p, qz_v_p,z_protein = z_encoder_p(encoder_input_2)

    return Float32.(z_gex),Float32.(z_protein), Float32.(qz_m), Float32.(qz_v),Float32.(qz_m_p), Float32.(qz_v_p), ql_m, ql_v, Float32.(library)
end 

function generative(::Type{scMultiVAE_},m, z::AbstractVector{Matrix{S}},library::AbstractMatrix{S}) where S <: Real
    
    z_rna, z_protein = z[1], z[2]
    # unpack decoders 
    decoder_g, decoder_p = m.decoder[1], m.decoder[2]
    px_scale, px_r, px_rate, px_dropout = decoder_g(z_rna, library)
    #if m.dispersion == :gene # some other cases (:gene-batch, :gene-label ignored)
    #    px_r = m.px_r
    #end
    px_r = exp.(px_r)
    px_scale_, px_r_, px_rate_, px_dropout_ = decoder_p(z_protein, library)
    px_r_ = exp.(px_r_)
    return px_scale, px_r, px_rate, px_dropout, px_scale_, px_r_, px_rate_, px_dropout_
end

function get_mixlatent_representation(model::scMultiVAE_, multiadata::AnnData; cellindices=nothing, give_mean::Bool=true)
    # set the model to test mode 
    testmode!(model, true)
    # countmatrix assumes cells x genes
    # TODO this can be parameterized; namely, we pass the # of GEX & protein instead of hardcoding 
    if !isnothing(cellindices)
        mod1 = multiadata.countmatrix[cellindices,1:4000]
        mod2 = multiadata.countmatrix[cellindices,4001:4134]
    end
    # pack the modalities countmatrices in an array 
    mod1 = multiadata.countmatrix[:,1:4000]
    mod2 = multiadata.countmatrix[:,4001:4134]
    x = [Float32.(mod1'), Float32.(mod2')]
    z_rna ,z_protein, qz_m_rna, qz_v_rna,qz_m_p, qz_v_p, ql_m, ql_v, library = inference(typeof(model),model,x)
    
    
    if give_mean
        # compute the mean, divide by 2 since we have 2 modalities
        mean_lats = (qz_m_rna' .+ qz_m_p') ./ 2
        return qz_m_rna, qz_m_p, mean_lats'
    else
        mean_z = (z_rna' .+ z_protein') ./ 2
        return z_rna, z_protein, mean_z'
    end
end
