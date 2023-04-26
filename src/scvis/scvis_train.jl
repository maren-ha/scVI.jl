# high-level function to dispatch on mode argument 
"""
Trains an scVAE model with the given AnnData, TrainingArgs and mode.

Parameters:
- m::scVAE: the scVAE model to be trained
- adata::AnnData: the AnnData object containing the dataset
- training_args::TrainingArgs: the TrainingArgs object containing the training parameters such as batch size, learning rate, etc.
- mode::Symbol: the training mode, can be one of the following:
    `:alternating`
    `:alternating_diff`
    `:joint`
    `:joint_diff`

Returns:
- Nothing. The model is trained in place.

Raises:
- A warning if an unsupported training mode is selected.
"""
function train_scvis_model!(m::scVAE, adata::AnnData, training_args::TrainingArgs, mode::Symbol)
    if !(mode ∈ [:alternating, :alternating_diff, :joint, :joint_diff])
        @warn "unsupported training mode selected, currently supported modes are 
        `:alternating`,
        `:alternating_diff`, 
        `:joint` and 
        `:joint_diff`"
    end
    return train_scvis_model!(m, adata, training_args, Val(mode))
end

#-------------------------------------------------------------------------------------------------------
# Version 1/2: joint training w/o differentiating through the similarity matrix 
#-------------------------------------------------------------------------------------------------------
function register_scvis_losses!(m::scVAE, x::AbstractMatrix{S}; kl_weight::Float32=1.0f0, epoch::Int=1) where S <: Real 
    encoder_input = m.log_variational ? log.(one(S) .+ x) : x
    q = m.z_encoder.encoder(x)
    q_m = m.z_encoder.mean_encoder(q)
    q_v = m.z_encoder.var_activation.(m.z_encoder.var_encoder(q)) .+ m.z_encoder.var_eps
    z = m.z_encoder.z_transformation(scVI.reparameterize_gaussian(q_m, q_v))
    ql_m, ql_v = nothing, nothing
    library = scVI.get_library(m, x, encoder_input)
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

    push!(m.loss_registry["kl_z"], mean(kl_divergence_z))
    push!(m.loss_registry["kl_l"], mean(kl_divergence_l))
    push!(m.loss_registry["kl_tsne"], mean(kl_qp))
    push!(m.loss_registry["reconstruction"], mean(reconst_loss))
    push!(m.loss_registry["total_loss"], lossval)
    return m
end

"""
Train an scVI model in joint mode.

Parameters:
- m: scVAE model to be trained
- adata: AnnData object containing the data to be used for training
- training_args: TrainingArgs object containing the training hyperparameters, such as weight decay, learning rate, batch size, and maximum number of epochs
- ::Val{:joint}: Symbol specifying the mode of training, must be :joint

Returns:
- Tuple of trained scVAE model, AnnData object

"""
function train_scvis_model!(m::scVAE, adata::AnnData, training_args::TrainingArgs, ::Val{:joint})

    opt = Flux.Optimiser(Flux.Optimise.WeightDecay(training_args.weight_decay), ADAM(training_args.lr))
    ps = Flux.params(m)

    ncells, ngenes = size(adata.X)

    if training_args.train_test_split
        trainsize = training_args.trainsize
        validationsize = 1 - trainsize # nothing 
        train_inds = shuffle!(collect(1:ncells))[1:Int(ceil(trainsize*ncells))]
    else
        train_inds = collect(1:ncells);
    end

    if training_args.register_losses
        m.loss_registry["kl_l"] = []
        m.loss_registry["kl_z"] = []
        m.loss_registry["kl_tsne"] = []
        m.loss_registry["reconstruction"] = []
        m.loss_registry["total_loss"] = []
    end

    #train_inds = shuffle(train_inds)
    #data = collect(adata.X[inds,:]' for inds in Iterators.partition(train_inds, training_args.batchsize))
    #Ps = collect(Float32.(compute_transition_probs(rescale(adata.X[inds,:]), 30.0)) for inds in Iterators.partition(train_inds, training_args.batchsize))
    #Ps = collect(compute_transition_probs(scVI.prcomps(adata.X[inds,:])[:,1:100]) for inds in Iterators.partition(train_inds, training_args.batchsize))
    #dataloader = zip(data, Ps)

    dataloader = Flux.DataLoader(adata.X[train_inds,:]', batchsize=training_args.batchsize, shuffle=true)

    train_steps=0
    @info "Starting training for $(training_args.max_epochs) epochs..."
    progress = training_args.progress && Progress(length(dataloader)*training_args.max_epochs);

    for epoch in 1:training_args.max_epochs
        kl_weight = scVI.get_kl_weight(training_args.n_epochs_kl_warmup, training_args.n_steps_kl_warmup, epoch, 0)
        for d in dataloader
            encoder_input = m.log_variational ? log.(one(eltype(d)) .+ d) : d
            curP = compute_transition_probs(m.z_encoder.encoder(encoder_input))
            curloss, back = Flux.pullback(ps) do 
                scvis_loss(m, d, curP; kl_weight=kl_weight, epoch=epoch)
            end
            grad = back(1f0)
            Flux.Optimise.update!(opt, ps, grad)
            if training_args.progress
                next!(progress; showvalues=[(:loss, curloss)]) 
            elseif (train_steps % training_args.verbose_freq == 0) && training_args.verbose
                @info "epoch $(epoch):" loss=curloss
            end
            if (epoch+1)%10 == 0
                training_args.register_losses && register_scvis_losses!(m, d, curP; kl_weight=kl_weight, epoch=epoch)
            end
            train_steps += 1
        end
    end
    @info "training complete!"
    m.is_trained = true
    #adata.is_trained = true
    return m, adata
end

function train_scvis_model!(m::scVAE, adata::AnnData, training_args::TrainingArgs, ::Val{:joint_diff})

    opt = Flux.Optimiser(Flux.Optimise.WeightDecay(training_args.weight_decay), ADAM(training_args.lr))
    ps = Flux.params(m)

    ncells, ngenes = size(adata.X)

    if training_args.train_test_split
        trainsize = training_args.trainsize
        validationsize = 1 - trainsize # nothing 
        train_inds = shuffle!(collect(1:ncells))[1:Int(ceil(trainsize*ncells))]
    else
        train_inds = collect(1:ncells);
    end

    if training_args.register_losses
        m.loss_registry["kl_l"] = []
        m.loss_registry["kl_z"] = []
        m.loss_registry["kl_tsne"] = []
        m.loss_registry["reconstruction"] = []
        m.loss_registry["total_loss"] = []
    end

    dataloader = Flux.DataLoader(adata.X[train_inds,:]', batchsize=training_args.batchsize, shuffle=true)

    train_steps=0
    @info "Starting training for $(training_args.max_epochs) epochs..."
    progress = training_args.progress && Progress(length(dataloader)*training_args.max_epochs);

    for epoch in 1:training_args.max_epochs
        kl_weight = scVI.get_kl_weight(training_args.n_epochs_kl_warmup, training_args.n_steps_kl_warmup, epoch, 0)
        for d in dataloader
            curloss, back = Flux.pullback(ps) do 
                scvis_loss(m, d, nothing; kl_weight=kl_weight, epoch=epoch)
            end
            grad = back(1f0)
            Flux.Optimise.update!(opt, ps, grad)
            if training_args.progress
                next!(progress; showvalues=[(:loss, curloss)]) 
            elseif (train_steps % training_args.verbose_freq == 0) && training_args.verbose
                @info "epoch $(epoch):" loss=curloss
            end
            if (epoch+1)%10 == 0
                training_args.register_losses && register_scvis_losses!(m, d, nothing; kl_weight=kl_weight, epoch=epoch)
            end
            train_steps += 1
        end
    end
    @info "training complete!"
    m.is_trained = true
    #adata.is_trained = true
    return m, adata
end

#-------------------------------------------------------------------------------------------------------
# Version 3/4: alternating training with separate tsne_net to go from 10 to 2-dim, w/o differentiating through the similarity matrix 
#-------------------------------------------------------------------------------------------------------

function Flux.params(m::scVAE, tsne_net::Dense)
    if !isnothing(m.l_encoder)
        ps = Flux.params(
            m.z_encoder.encoder, m.z_encoder.mean_encoder, m.z_encoder.var_encoder,
            m.l_encoder.encoder, m.l_encoder.mean_encoder, m.l_encoder.var_encoder,
            tsne_net
        );
    else
        ps = Flux.params(
            m.z_encoder.encoder, m.z_encoder.mean_encoder, m.z_encoder.var_encoder,
            tsne_net
        );
    end
    return ps
end

"""
    This function register the losses for scVAE (scVI Variational Autoencoder) model.
    It calculates the KL divergence, reconstruction loss, and tsne loss, and adds them to the loss_registry of the model.
    
    Parameters:
    - m: scVAE model
    - tsne_net: Dense network for t-SNE calculation
    - x: Input data, matrix of size (batchsize, number of genes)
    - kl_weight: weight for the KL divergence loss, default to 1.0f0
    - epoch: current epoch number, used for computing tsne loss, default to 1
    
    Returns:
    - m: scVAE model with updated `loss_registry``
    
    Example:
    m = scVAE(...)
    tsne_net = Dense(...)
    x = ...
    register_scvis_losses!(m, tsne_net, x)
    """
function register_scvis_losses!(m::scVAE, tsne_net::Dense, x::AbstractMatrix{S}; kl_weight::Float32=1.0f0, epoch::Int=1) where S <: Real 
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
    lossval = mean(reconst_loss + weighted_kl_local)

    # tsne part 
    latent = z # alternative: qz_m? 
    z_tsne = tsne_net(latent)
    P = compute_differentiable_transition_probs(latent)
    kl_qp = tsne_repel(z_tsne, P) * min(epoch, m.n_latent)

    push!(m.loss_registry["kl_z"], mean(kl_divergence_z))
    push!(m.loss_registry["kl_l"], mean(kl_divergence_l))
    push!(m.loss_registry["kl_tsne"], kl_qp)
    push!(m.loss_registry["reconstruction"], mean(reconst_loss))
    push!(m.loss_registry["total_loss"], lossval)
    return m
end

function train_scvis_model!(m::scVAE, adata::AnnData, training_args::TrainingArgs, ::Val{:alternating}; seed::Int=42)

    Random.seed!(seed)
    tsne_net = Dense(m.n_latent, 2)

    opt = Flux.Optimiser(Flux.Optimise.WeightDecay(training_args.weight_decay), ADAM(training_args.lr))
    ps = Flux.params(m)
    tsne_opt = Flux.Optimiser(Flux.Optimise.WeightDecay(training_args.weight_decay), ADAM(training_args.lr))
    tsne_ps = Flux.params(m, tsne_net)

    ncells, ngenes = size(adata.X)

    if training_args.train_test_split
        trainsize = training_args.trainsize
        validationsize = 1 - trainsize # nothing 
        train_inds = shuffle!(collect(1:ncells))[1:Int(ceil(trainsize*ncells))]
    else
        train_inds = collect(1:ncells);
    end

    if training_args.register_losses
        m.loss_registry["kl_l"] = []
        m.loss_registry["kl_z"] = []
        m.loss_registry["kl_tsne"] = []
        m.loss_registry["reconstruction"] = []
        m.loss_registry["total_loss"] = []
    end

    dataloader = Flux.DataLoader(adata.X[train_inds,:]', batchsize=training_args.batchsize, shuffle=true)

    train_steps=0
    @info "Starting training for $(training_args.max_epochs) epochs..."
    progress = training_args.progress && Progress(length(dataloader)*training_args.max_epochs);

    for epoch in 1:training_args.max_epochs
        kl_weight = scVI.get_kl_weight(training_args.n_epochs_kl_warmup, training_args.n_steps_kl_warmup, epoch, 0)
        for d in dataloader
            # VAE update 
            #@info "tick"
            curloss, back = Flux.pullback(ps) do 
                scVI.loss(m, d; kl_weight=kl_weight)    
            end
            grad = back(1f0)
            Flux.Optimise.update!(opt, ps, grad)
            # tsne net + encoder update
            #@info "tack"
            z, qz_m, qz_v, ql_m, ql_v, library = scVI.inference(m, d)
            P = compute_transition_probs(z)
            tsne_curloss, tsne_back = Flux.pullback(tsne_ps) do 
                tsne_loss(m, tsne_net, d, P, epoch)
            end
            tsne_grad = tsne_back(1f0)
            Flux.Optimise.update!(tsne_opt, tsne_ps, tsne_grad)
            #@info "tock"
            if training_args.progress
                next!(progress; showvalues=[(:loss, curloss), (:tsne_loss, tsne_curloss)]) 
            elseif (train_steps % training_args.verbose_freq == 0) && training_args.verbose
                @info "epoch $(epoch):" loss=curloss tsne_loss=tsne_curloss
            end
            if (epoch-1)%10 == 0
                training_args.register_losses && register_losses!(m, tsne_net, d; kl_weight=kl_weight, epoch=epoch)
            end
            train_steps += 1
        end
    end
    @info "training complete!"
    m.is_trained = true
    #adata.is_trained = true
    return m, tsne_net
end

function train_scvis_model!(m::scVAE, adata::AnnData, training_args::TrainingArgs, ::Val{:alternating_diff}; seed::Int=42)

    Random.seed!(seed)
    tsne_net = Dense(m.n_latent, 2)

    opt = Flux.Optimiser(Flux.Optimise.WeightDecay(training_args.weight_decay), ADAM(training_args.lr))
    ps = Flux.params(m)
    tsne_opt = Flux.Optimiser(Flux.Optimise.WeightDecay(training_args.weight_decay), ADAM(training_args.lr))
    tsne_ps = Flux.params(m, tsne_net)

    ncells, ngenes = size(adata.X)

    if training_args.train_test_split
        trainsize = training_args.trainsize
        validationsize = 1 - trainsize # nothing 
        train_inds = shuffle!(collect(1:ncells))[1:Int(ceil(trainsize*ncells))]
    else
        train_inds = collect(1:ncells);
    end

    if training_args.register_losses
        m.loss_registry["kl_l"] = []
        m.loss_registry["kl_z"] = []
        m.loss_registry["kl_tsne"] = []
        m.loss_registry["reconstruction"] = []
        m.loss_registry["total_loss"] = []
    end

    dataloader = Flux.DataLoader(adata.X[train_inds,:]', batchsize=training_args.batchsize, shuffle=true)

    train_steps=0
    @info "Starting training for $(training_args.max_epochs) epochs..."
    progress = training_args.progress && Progress(length(dataloader)*training_args.max_epochs);

    for epoch in 1:training_args.max_epochs
        kl_weight = scVI.get_kl_weight(training_args.n_epochs_kl_warmup, training_args.n_steps_kl_warmup, epoch, 0)
        for d in dataloader
            # VAE update 
            curloss, back = Flux.pullback(ps) do 
                scVI.loss(m, d; kl_weight=kl_weight)    
            end
            grad = back(1f0)
            Flux.Optimise.update!(opt, ps, grad)
            # tsne net + encoder update
            tsne_curloss, tsne_back = Flux.pullback(tsne_ps) do 
                tsne_loss(m, tsne_net, d, nothing, epoch)
            end
            tsne_grad = tsne_back(1f0)
            Flux.Optimise.update!(tsne_opt, tsne_ps, tsne_grad)
            if training_args.progress
                next!(progress; showvalues=[(:loss, curloss), (:tsne_loss, tsne_curloss)]) 
            elseif (train_steps % training_args.verbose_freq == 0) && training_args.verbose
                @info "epoch $(epoch):" loss=curloss tsne_loss=tsne_curloss
            end
            if (epoch-1)%10 == 0
                training_args.register_losses && register_losses!(m, tsne_net, d; kl_weight=kl_weight, epoch=epoch)
            end
            train_steps += 1
        end
    end
    @info "training complete!"
    m.is_trained = true
    #adata.is_trained = true
    return m, tsne_net
end

