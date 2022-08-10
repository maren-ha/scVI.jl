Base.@kwdef mutable struct TrainingArgs
    trainsize::Float32 = 0.9f0
    train_test_split::Bool=false
    batchsize::Int=64
    max_epochs::Int = 20
    lr::Float64 = 1e-3
    weight_decay::Float32 = 0.0f0
    n_steps_kl_warmup = nothing
    n_epochs_kl_warmup::Int=1
    progress::Bool=false
    verbose::Bool=true
    verbose_freq::Int=0
    log_path::String
end

function start_training(m,adata::AbstractArray{AnnData}, training_args::TrainingArgs,logger)
    # values to be returned m , adata 
    m , adata = train_model!(typeof(m),m, adata,training_args,logger)
    return m, adata 
end

Flux.params(m::scVAE) = Flux.params(m.z_encoder, m.l_encoder, m.decoder)
function train_model!(::Type{scVAE},m, adata::AbstractArray{AnnData}, training_args::TrainingArgs,logger)
    opt = Flux.Optimiser(Flux.Optimise.WeightDecay(training_args.weight_decay), ADAM(training_args.lr))
    ps = Flux.params(m)

    # unpack the data 
    adata = adata[1]
    if training_args.train_test_split
        trainsize = training_args.trainsize
        validationsize = 1 - trainsize # nothing 
        train_inds = shuffle!(collect(1:adata.ncells))[1:Int(ceil(trainsize*adata.ncells))]
    else
        train_inds = collect(1:adata.ncells);
    end

    dataloader = Flux.DataLoader(adata.countmatrix[train_inds,:]', batchsize=training_args.batchsize, shuffle=true)

    train_steps=0
    @info "Starting training for $(training_args.max_epochs) epochs..."
    progress = training_args.progress && Progress(length(dataloader)*training_args.max_epochs);

    for epoch in 1:training_args.max_epochs
        kl_weight = get_kl_weight(training_args.n_epochs_kl_warmup, training_args.n_steps_kl_warmup, epoch, 0)
        for d in dataloader
            curloss, back = Flux.pullback(ps) do 
                loss(m, d; kl_weight=kl_weight)    
            end
            grad = back(1f0)
            Flux.Optimise.update!(opt, ps, grad)
            if training_args.progress
                next!(progress; showvalues=[(:loss, curloss)])   
            elseif (train_steps % training_args.verbose_freq == 0) && training_args.verbose
                @info "epoch $(epoch):" loss=curloss
                with_logger(logger) do
                    @info string("training_loss") Total_loss=curloss
                
                end
            end
            train_steps += 1
            # calling garbage collectore
            
        end
    end
    @info "training complete!"
    adata.is_trained = true
    return m, adata
end

##########################################################################################################################
Flux.params(m::scMultiVAE_) = Flux.params( m.z_encoder[1],m.z_encoder[2], m.l_encoder, m.decoder[1], m.l_encoder_, m.decoder[2])


function train_model!(::Type{scMultiVAE_},m,adata::AbstractArray{AnnData}, training_args::TrainingArgs,logger)

    opt = Flux.Optimiser(Flux.Optimise.WeightDecay(training_args.weight_decay), ADAM(training_args.lr))
    ps = Flux.params(m)
    # unpack the data 
    adata1 , adata2 = adata[1], adata[2]
    if training_args.train_test_split
        trainsize = training_args.trainsize
        validationsize = 1 - trainsize # nothing 
        train_inds = shuffle!(collect(1:adata1.ncells))[1:Int(ceil(trainsize*adata1.ncells))]
    else
        train_inds_rna = collect(1:adata1.ncells);
        train_inds_protein = collect(1:adata2.ncells);
    end

    dataloader_mod1 = Flux.DataLoader(adata1.countmatrix[train_inds_rna,:]', batchsize=training_args.batchsize, shuffle=true)
    dataloader_mod2 = Flux.DataLoader(adata2.countmatrix[train_inds_protein,:]', batchsize=training_args.batchsize, shuffle=true)

    train_steps=0
    @info "Starting training for $(training_args.max_epochs) epochs..."
    progress = training_args.progress && Progress(length(dataloader_mod1)*training_args.max_epochs);
    for epoch in 1:training_args.max_epochs
        kl_weight = get_kl_weight(training_args.n_epochs_kl_warmup, training_args.n_steps_kl_warmup, epoch, 0)
        for (mod1,mod2) in zip(dataloader_mod1,dataloader_mod2)
            curloss, back = Flux.pullback(ps) do 
                loss(m, [Float32.(mod1),Float32.(mod2)], kl_weight=kl_weight)   
            end
            grad = back(1f0)
            Flux.Optimise.update!(opt, ps, grad)
            if training_args.progress
                next!(progress; showvalues=[(:loss, curloss)])
                # this call back for our sanity check ... 
                total, recon_rna , recon_pro = objective_multimodal_callback(m, mod1, mod2, kl_weight=kl_weight)
                @info "epoch $(epoch):" recon_rna=recon_rna
                @info "epoch $(epoch):" recon_pro=recon_pro
                with_logger(logger) do
                    @info string("Training_Loss") Total_loss=curloss
                    @info string("RNA_Loss") RNA_loss=recon_rna
                    @info string("Protein_Loss") Protein_loss=recon_pro
                end
            elseif (train_steps % training_args.verbose_freq == 0) && training_args.verbose
                total, recon_rna , recon_pro   = objective_multimodal_callback(m, mod1, mod2, kl_weight=kl_weight)
                @info "epoch $(epoch):" total=total
                @info "epoch $(epoch):" recon_rna=recon_rna
                @info "epoch $(epoch):" recon_pro=recon_pro
                with_logger(logger) do
                    @info string("Training_Loss") Total_loss=curloss
                    @info string("RNA_Loss") RNA_loss=recon_rna
                    @info string("Protein_Loss") Protein_loss=recon_pro
                end
            end
            train_steps += 1
        end

    end
    @info "training complete!"
    adata1.is_trained = true
    adata2.is_trained = true
    adata = [adata1, adata2]
    return m, adata
end


function objective_multimodal_callback(m::scMultiVAE_, mod1::AbstractMatrix{S}, mod2::AbstractMatrix{S}; kl_weight::Float32=1.0f0) where S <: Real 
    #apply inference on multimodality
    x = [mod1, mod2] 
    z_rna ,z_protein, qz_m_rna, qz_v_rna,qz_m_p, qz_v_p, ql_m, ql_v, library  = do_inference(m, x)
    z = [z_rna, z_protein]
    #apply generative on multimodal
    px_scale, px_r, px_rate, px_dropout, px_scale_, px_r_, px_rate_, px_dropout_ = do_generative(m,z,library)
        
    # no off diagonal calculation
    kl_divergence_rna = -0.5f0 .* sum(1.0f0 .+ log.(qz_v_rna) - qz_m_rna.^2 .- qz_v_rna, dims=1) # batchxlatent_dimentionality

    kl_divergence_p = -0.5f0 .* sum(1.0f0 .+ log.(qz_v_p) - qz_m_p.^2 .- qz_v_p, dims=1)
    # compute reconstruction error for rna modality 
    
    reconstruction_loss_rna =  sum(-log_zinb_positive(Float32.(mod1), Float32.(px_rate), Float32.(px_r), Float32.(px_dropout)), dims=1)
    # compute reconstruction error for protein modality 
    # px_z
    reconstruction_loss_p = sum(-log_nb_positive(Float32.(mod2), Float32.(px_rate_), Float32.(px_r_)), dims=1)
    # total reconstraction loss 
    recon_loss = sum(reconstruction_loss_p) + sum(reconstruction_loss_rna)
    kl_loss = sum(kl_divergence_rna) + sum(kl_divergence_p)
    multimodal_total_loss = sum(0.5 * (recon_loss + (kl_weight .* kl_loss)))
    return mean(multimodal_total_loss) , mean(reconstruction_loss_rna), mean(reconstruction_loss_p)
end