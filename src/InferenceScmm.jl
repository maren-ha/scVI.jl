##########################################this is for scmm###################################################################

function scmminference(m::scmmVAE, x::AbstractMatrix{S}) where S <: Real 
    encoder_input = m.log_variational ? log.(one(S) .+ x) : x
    qz_m, qz_v = m.encoder(encoder_input)
    
    z = reparameterize_gaussian(qz_m, qz_v)
    
    return z, qz_m, qz_v
end

function scmmgenerative(m::scmmVAE, z::AbstractMatrix{S}) where S <: Real 
    decoder_input = z
    px_r, px_rate = m.decoder(decoder_input)
    #px_r = exp.(px_r)
    return  px_r, px_rate
end



function loss_(m::scmmVAE, x::AbstractMatrix{S}; kl_weight::Float32=1.0f0) where S <: Real 
    z, qz_m, qz_v = scmminference(m, x)
    px_r, px_rate = scmmgenerative(m, z)
    kl_divergence_z = -0.5f0 .* sum(1.0f0 .+ log.(qz_v) - qz_m.^2 .- qz_v, dims=1) # 2 
    # equivalent to kl_divergence_z = torch.distributions.kl.kl_divergence(torch.distributions.Normal(qz_m, qz_v.sqrt()), torch.distributions.Normal(mean, scale)).sum(dim=1)
    kl_divergence_l = 0.0f0
    reconst_loss = sum(-log_nb_positive(Float32.(x), Float32.(px_rate), Float32.(px_r)), dims=1)
    kl_local_for_warmup = kl_divergence_z
    kl_local_no_warmup = kl_divergence_l
    weighted_kl_local = kl_weight .* kl_local_for_warmup .+ kl_local_no_warmup

    lossval = mean(reconst_loss + weighted_kl_local)
    return lossval
end