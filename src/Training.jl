"""mutable struct TrainingArgs

Struct to store hyperparameters to control and customise the training process of an `scVAE` or `scMultiVAE_` model. 
Can be constructed using keywords. 

**Keyword arguments:**
----------------------
 - `trainsize::Float32=0.9f0`: proportion of data to be used for training when using a train-test split for training. Has no effect when `train_test_split==false`.
 - `train_test_split::Bool=false`: whether or not to randomly split the data into training and test set. 
 - `batchsize::Int=128`: batchsize to be used when partitioning the data into minibatches for training based on stochastic gradient descent 
 - `max_epochs::Int=400`: number of epochs to train the model 
 - `lr::Float64=1e-3`: learning rate (=stepsize) of the ADAM optimiser during the stochastic descent optimisation for model training (for details, see `?ADAM`). 
 - `weight_decay::Float32=0.0f0`: rate of weight decay to apply in the ADAM optimiser (for details, see `?ADAM`).
 - `n_steps_kl_warmup::Union{Int, Nothing}=nothing`: number of steps (one gradient descent optimiser update for one batch) over which to perform gradual increase (warm-up, annealing) of the weight of the regularising KL-divergence term in the loss function (ensuring the consistency between variational posterior and standard normal prior). Empirically, this improves model inference.
 - `n_epochs_kl_warmup::Union{Int, Nothing}=400`: number of epochs (one update for all batches) over which to perform gradual increase (warm-up, annealing) of the weight of the regularising KL-divergence term in the loss function (ensuring the consistency between variational posterior and standard normal prior). Empirically, this improves model inference.
 - `progress::Bool=true`: whether or not to print a progress bar and the current value of the loss function to the REPL.
 - `register_losses::Bool=false`: whether or not to record the values of the different loss components after each training epoch in the `loss_registry` of the `scVAE` model. If `true`, for each loss component (reconstruction error, KL divergences, total loss), an array will be created in the dictionary with the name of the loss component as key, where after each epoch, the value of the component is saved.
 - `verbose::Bool=false`: only kicks in if `progress==false`: whether or not to print the current epoch and value of the loss function every `verbose_freq` epoch. 
 - `infotime::Int=1`: frequency with which to display the current epoch and current value of the loss function (only if epoch mod infotime==0 ).
 - `tblogger::Bool=true`: if true the losses will be logged on a tensorboard.
 - `savepath::String=nothing`: directory to save the model. 
 - `use_cuda::Bool=false`: moves the training to a GPU if available.
 - `checktime::Int=5`: the model will be saved if `args.checktime > 0 && epoch % args.checktime == 0`
"""
Base.@kwdef mutable struct TrainingArgs
    trainsize::Float32 = 0.9f0
    train_test_split::Bool=false
    batchsize::Int=128
    max_epochs::Int = 50
    lr::Float64 = 1e-3
    weight_decay::Float32 = 0.0f0
    n_steps_kl_warmup::Union{Int, Nothing}=nothing
    n_epochs_kl_warmup::Union{Int, Nothing}=400
    register_losses::Bool=true
    progress::Bool=false
    verbose::Bool=true
    infotime::Int=1
    tblogger::Bool=true
    savepath::String=nothing
    use_cuda::Bool=false
    seed::Int=123
    checktime::Int=5
    train_w_tsne::Bool=false
end

function start_training!(m,adata::AbstractArray{AnnData}, training_args::TrainingArgs,logger)
    # values to be returned m , adata, tuple of losses for plotting as png image. 
    # This is good for the ones who do not have tensorboard installed
    m , adata, losses = train_model!(typeof(m),m, adata,training_args,logger)
    return m, adata, losses
end
"""
    train_model!(m::scVAE, adata::AnnData, training_args::TrainingArgs)

Trains an `scVAE` model on an `AnnData` object, where the behaviour is controlled by a `TrainingArgs` object: 
Defines the ADAM SGD optimiser, collects the model parameters, optionally splits data in training and testdata and 
initialises a `Flux.DataLoader` storing the data in the countmatrix of the `AnnData` object in batches. 
Updates the model parameters via stochastic gradient for the specified number of epochs, 
optionally prints out progress and current loss values. 

Returns the trained `scVAE` model.
"""
Flux.params(m::scVAE) = Flux.params(m.z_encoder, m.l_encoder, m.decoder)
function train_model!(::Type{scVAE},m, adata::AbstractArray{AnnData}, training_args::TrainingArgs,logger)
    args = training_args
    # create seed ....
    args.seed > 0 && Random.seed!(args.seed)
    use_cuda = args.use_cuda && CUDA.functional()
    
    if use_cuda
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end
    adata = adata[1]
    ## move the model to a device ...
    model = m|> device
    opt = Flux.Optimiser(Flux.Optimise.WeightDecay(args.weight_decay), ADAM(args.lr))
    ps = Flux.params(model)
    
    @info "The ScVI has: $(num_params(model)) trainable params"    

    if args.train_test_split
        trainsize = args.trainsize
        validationsize = 1 - trainsize # nothing 
        train_inds = shuffle!(collect(1:adata.ncells))[1:Int(ceil(trainsize*adata.ncells))]
    else
        train_inds = collect(1:adata.ncells);
    end

    tsne_turn = false 
    if args.register_losses
        model.loss_registry["kl_l"] = []
        model.loss_registry["kl_z"] = []
        model.loss_registry["reconstruction"] = []
        model.loss_registry["total_loss"] = []
    end
    # for plotting the losses after training, just sanity check! 
    kl_z = []
    recon_loss = []
    total_loss = []
    if model.train_w_tsne
        tsne_loss = []
    end
    ## Report losses ...
        
    function report(epoch,kl_weight) 
        
        train = objective_unimodel_callback(model,adata.countmatrix[train_inds,:]',tsne_turn=false,kl_weight=kl_weight)
        push!(kl_z,train.kl_d)
        push!(recon_loss,train.recon_loss)
        push!(total_loss,train.total_loss)
            
        println("Epoch: $epoch   Train: $(train)")
            
        if args.tblogger
            set_step!(logger, epoch)
            with_logger(logger) do
                @info "train_reconstruction_error" reconstruction=train.recon_loss 
                @info "train_kl_divergence" kl_divergence=train.kl_d
                if tsne_turn
                    @info "train_tsne_loss" tsne_loss=train.tsne_loss
                end
                @info "train_total_loss" total_loss=train.total_loss

            end
        end
    end
    dataloader = Flux.DataLoader(adata.countmatrix[train_inds,:]', batchsize=args.batchsize, shuffle=true)
    ## START TRAINING
    train_steps=0
    @info "Starting training for $(training_args.max_epochs) epochs..."
    progress = args.progress && Progress(length(dataloader)*args.max_epochs);

    for epoch in 1:args.max_epochs
        kl_weight = get_kl_weight(args.n_epochs_kl_warmup, args.n_steps_kl_warmup, epoch, 0)
        for d in dataloader
            d = d |> device
            curloss, back = Flux.pullback(ps) do 
                loss(model, d; tsne_turn, kl_weight=kl_weight)    
            end
            grad = back(1f0)
            if args.progress
                next!(progress; showvalues=[(:loss, curloss)])  
            end
            Flux.Optimise.update!(opt, ps, grad)
            train_steps += 1
        end
        #TODO activate the registry from Maren!
        #args.register_losses && register_losses!(model, Float32.(adata.countmatrix[train_inds,:]'); kl_weight=kl_weight)
        
        ## Printing and logging
        epoch % args.infotime == 0 && report(epoch,kl_weight)
        if args.checktime > 0 && epoch % args.checktime == 0
            !ispath(args.savepath) && mkpath(args.savepath)            
            modelpath = joinpath(args.savepath, "model.bson") 
            let model = cpu(model) ## return model to cpu before serialization
                  BSON.@save modelpath model epoch
            end
            @info "Model saved in \"$(modelpath)\""
        end        
    end
    @info "training complete!"
    model.is_trained = true
    if model.train_w_tsne #TODO in the coming comits we will have tsne loss added
        return model, adata ,(total_loss=total_loss, reconstruction_loss=recon_loss, kl_d=kl_z,tsne_loss=0)
    else
        return model, adata ,(total_loss=total_loss, reconstruction_loss=recon_loss, kl_d=kl_z)
    end
end
#############################################
#TODO this fuction for reporting while training, I will move it to train utils in the coming commits
function objective_unimodel_callback(m::scVAE, x::AbstractMatrix{S}; tsne_turn=false,kl_weight::Float32=1.0f0) where S <: Real 
        # pack the data in an array 
        x = copy(x)# to get rid of adjoint
        x = [x] 
        z, qz_m, qz_v, ql_m, ql_v, library, z_tsne = do_inference(m, x,tsne_turn)
        z = copy(z)# to get rid of adjoint
        z = [z] 
        px_scale, px_r, px_rate, px_dropout = do_generative(m, z, library,tsne_turn)
        kl_divergence_z = -0.5f0 .* sum(1.0f0 .+ log.(qz_v) - qz_m.^2 .- qz_v, dims=1) # 2 
        # equivalent to kl_divergence_z = torch.distributions.kl.kl_divergence(torch.distributions.Normal(qz_m, qz_v.sqrt()), torch.distributions.Normal(mean, scale)).sum(dim=1)
    
        if !m.use_observed_lib_size # TODO!
            local_library_log_means,local_library_log_vars = _compute_local_library_params() 
            kl_divergence_l = kl(Normal(ql_m, ql_v.sqrt()),Normal(local_library_log_means, local_library_log_vars.sqrt())).sum(dim=1)
        else
            kl_divergence_l = 0.0f0
        end
        d = x[1]
        reconst_loss = get_reconstruction_loss(m, copy(Float32.(d)), Float32.(px_rate), Float32.(px_r), Float32.(px_dropout))
        kl_local_for_warmup = kl_divergence_z
        kl_local_no_warmup = kl_divergence_l
        weighted_kl_local = kl_weight .* kl_local_for_warmup .+ kl_local_no_warmup
    
        #TODO add the tsne loss
        if tsne_turn & m.train_w_tsne
            # calculate tsne loss 
        end 
        lossval = mean(reconst_loss + weighted_kl_local)
        #kl_local = Dict("kl_divergence_l" => kl_divergence_l, "kl_divergence_z" => kl_divergence_z)
        #kl_global = [0.0]
        #return lossval#, reconst_loss, kl_local, kl_global
        return (total_loss=lossval |> round4 , recon_loss=mean(reconst_loss) |> round4, kl_d=mean(kl_divergence_z) |> round4)
end
##########################################################################################################################
#TODO make this ugly implementation nicer abnd move it to a train utils file 
Flux.params(m::scMultiVAE_) = Flux.params(m.z_encoder[1],m.z_encoder[2], m.l_encoder, m.decoder[1], m.l_encoder_, m.decoder[2])
function objective_multimodal_callback(m::scMultiVAE_, mod1::AbstractMatrix{S}, mod2::AbstractMatrix{S}; kl_weight::Float32=1.0f0) where S <: Real 
    #apply inference on multimodality
    x = [Float32.(mod1),Float32.(mod2)]
    mod1, mod2 = Float32.(mod1),Float32.(mod2)
    z_rna ,z_protein, qz_m_rna, qz_v_rna,qz_m_p, qz_v_p, ql_m, ql_v, library  = do_inference(m, x)
    z = [z_rna, z_protein]
    #apply generative on multimodal
    px_scale, px_r, px_rate, px_dropout, px_scale_, px_r_, px_rate_, px_dropout_ = do_generative(m,z,library)
    # no off diagonal calculation
    kl_divergence_rna = -0.5f0 .* sum(1.0f0 .+ log.(qz_v_rna) - qz_m_rna.^2 .- qz_v_rna, dims=1) # batchxlatent
    kl_divergence_p = -0.5f0 .* sum(1.0f0 .+ log.(qz_v_p) - qz_m_p.^2 .- qz_v_p, dims=1)
    # compute reconstruction error for rna modality 
    reconstruction_loss_rna =  sum(-log_zinb_positive(Float32.(mod1), Float32.(px_rate), Float32.(px_r), Float32.(px_dropout)), dims=1)
    # compute reconstruction error for protein modality 
    # px_z
    reconstruction_loss_p = sum(-log_nb_positive(Float32.(mod2), Float32.(px_rate_), Float32.(px_r_)), dims=1)
    # total reconstraction loss 
    recon_loss = sum(reconstruction_loss_p) + sum(reconstruction_loss_rna)
    kl_loss = sum(kl_divergence_rna) + sum(kl_divergence_p)
    multimodal_total_loss = sum(0.5 * (recon_loss - (kl_weight .* kl_loss)))    
    return (moe_loss=mean(multimodal_total_loss) |> round4 , loss_rna=mean(reconstruction_loss_rna) |> round4, loss_protein=mean(reconstruction_loss_p) |> round4)
end

## Utility functions
num_params(model) = sum(length, Flux.params(model))
round4(x) = round(x, digits=4)
function train_model!(::Type{scMultiVAE_},m,multi_adata::AbstractArray{AnnData}, training_args::TrainingArgs,logger)
    args = training_args
    # create seed ....
    args.seed > 0 && Random.seed!(args.seed)
    use_cuda = args.use_cuda && CUDA.functional()
    
    if use_cuda
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    ## Get the data ... 
    multi_adata = multi_adata[1]

    mod1 = multi_adata.countmatrix[:,1:4000]
    mod2 = multi_adata.countmatrix[:,4001:4134]
    if  args.train_test_split
        trainsize = args.trainsize
        validationsize = 1 - trainsize # nothing 
        train_inds = shuffle!(collect(1: multi_adata.ncells))[1:Int(ceil(trainsize* multi_adata.ncells))]
        # train loaders 
        dataloader_mod1 = Flux.DataLoader(mod1[train_inds,:]', batchsize=training_args.batchsize)
        dataloader_mod2 = Flux.DataLoader(mod2[train_inds,:]', batchsize=training_args.batchsize)
        # Get non intersection indices set = validation indices       
        val_inds = collect(1: multi_adata.ncells)[collect(1: multi_adata.ncells) .∉ Ref(train_inds)]
        val_dataloader_mod1 = Flux.DataLoader(mod1[val_inds,:]', batchsize=training_args.batchsize)
        val_dataloader_mod2 = Flux.DataLoader(mod2[val_inds,:]', batchsize=training_args.batchsize)        
    else
        train_inds_rna = collect(1: multi_adata.ncells);
        train_inds_protein = collect(1: multi_adata.ncells);
        # get dataloaders for each modality ... 
        dataloader_mod1 = Flux.DataLoader(mod1[train_inds_rna,:]', batchsize=training_args.batchsize,shuffle=false,partial=false)
        dataloader_mod2 = Flux.DataLoader(mod2[train_inds_protein,:]', batchsize=training_args.batchsize,shuffle=false,partial=false)
    end

    ## move the model to a device ...
    model = m|> device

    @info "The MulitScVI has: $(num_params(model)) trainable params"    
    opt = Flux.Optimiser(Flux.Optimise.WeightDecay(args.weight_decay), ADAM(args.lr))
    
    ps = Flux.params(model)  
    
    moes_loss = []
    loss_rnas = []
    loss_proteins = []

    val_moes_loss = []
    val_loss_rnas = []
    val_loss_proteins = []

    ## Report losses ...
    function report(epoch,kl_weight)
        train = objective_multimodal_callback(model, mod1[train_inds_rna,:]', mod2[train_inds_protein,:]', kl_weight=kl_weight)
        push!(moes_loss,train.moe_loss)
        push!(loss_rnas,train.loss_rna)
        push!(loss_proteins,train.loss_protein)
        println("Epoch: $epoch   Train: $(train)")
        if args.tblogger
            set_step!(logger, epoch)
            with_logger(logger) do
                @info "train" moe_loss=train.moe_loss 
                @info "train_RNA" loss_rna=train.loss_rna  
                @info "train_Protein" loss_protein=train.loss_protein 
            end

        end
    end
    
    ## START TRAINING
    @info "Starting training for $(training_args.max_epochs) epochs..."
    progress = args.progress && Progress(length(dataloader_mod1)*args.max_epochs);
    report(0,0.0f0)
    for epoch in 1:args.max_epochs
        kl_weight = get_kl_weight(training_args.n_epochs_kl_warmup, training_args.n_steps_kl_warmup, epoch, 0)
        for (mod1_, mod2_) in zip(dataloader_mod1,dataloader_mod2)
            mod1_, mod2_ = mod1_ |> device, mod2_ |> device
            curloss, back = Flux.pullback(ps) do 
                loss(m, [Float32.(mod1_),Float32.(mod2_)], kl_weight=kl_weight)    
            end
            grad = back(1f0)
            if args.progress
                next!(progress; showvalues=[(:loss, curloss)])  
            end

            Flux.Optimise.update!(opt, ps, grad)
        end
        ## Printing and logging
        epoch % args.infotime == 0 && report(epoch,kl_weight)
        if args.checktime > 0 && epoch % args.checktime == 0
            
            !ispath(args.savepath) && mkpath(args.savepath)            
            if args.train_test_split
                validation = objective_multimodal_callback(model, mod1[val_inds,:]', mod2[val_inds,:]', kl_weight=kl_weight)
                push!(val_moes_loss,validation.moe_loss)
                push!(val_loss_rnas,validation.loss_rna)
                push!(val_loss_proteins,validation.loss_protein)
            end
            modelpath = joinpath(args.savepath, "model.bson") 
            let model = cpu(model) ## return model to cpu before serialization
                BSON.@save modelpath model epoch
            end
            @info "Model saved in \"$(modelpath)\""
        end
            
    end
    @info "training complete!"
    model.is_trained = true
    
    return model, multi_adata, 
    (moes_train=moes_loss, mod1_train=loss_rnas, mod2_train=loss_proteins, 
    moes_val=val_moes_loss, mod1_val=val_loss_rnas, mod2_val2=val_loss_proteins)
end