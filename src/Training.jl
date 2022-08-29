"""
    mutable struct TrainingArgs

Struct to store hyperparameters to control and customise the training process of an `scVAE` model. 
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
 - `verbose::Bool=false`: only kicks in if `progress==false`: whether or not to print the current epoch and value of the loss function every `verbose_freq` epoch. 
 - `verbose_freq::Int=10`: frequency with which to display the current epoch and current value of the loss function (only if `progress==false` and `verbose==true`).
"""
Base.@kwdef mutable struct TrainingArgs
    trainsize::Float32=0.9f0
    train_test_split::Bool=false
    batchsize::Int=128
    max_epochs::Int=400
    lr::Float64=1e-3
    weight_decay::Float32=0.0f0
    n_steps_kl_warmup::Union{Int, Nothing}=nothing
    n_epochs_kl_warmup::Union{Int, Nothing}=400
    register_losses::Bool=false
    progress::Bool=true
    verbose::Bool=false
    verbose_freq::Int=10
end

Flux.params(m::scVAE) = Flux.params(m.z_encoder, m.l_encoder, m.decoder)

"""
    train_model!(m::scVAE, adata::AnnData, training_args::TrainingArgs)

Trains an `scVAE` model on an `AnnData` object, where the behaviour is controlled by a `TrainingArgs` object: 
Defines the ADAM SGD optimiser, collects the model parameters, optionally splits data in training and testdata and 
initialises a `Flux.DataLoader` storing the data in the countmatrix of the `AnnData` object in batches. 
Updates the model parameters via stochastic gradient for the specified number of epochs, 
optionally prints out progress and current loss values. 

Returns the trained `scVAE` model.
"""
function train_model!(m::scVAE, adata::AnnData, training_args::TrainingArgs)

    opt = Flux.Optimiser(Flux.Optimise.WeightDecay(training_args.weight_decay), ADAM(training_args.lr))
    ps = Flux.params(m)

    if training_args.train_test_split
        trainsize = training_args.trainsize
        validationsize = 1 - trainsize # nothing 
        train_inds = shuffle!(collect(1:adata.ncells))[1:Int(ceil(trainsize*adata.ncells))]
    else
        train_inds = collect(1:adata.ncells);
    end

    if training_args.register_losses
        m.loss_registry["kl_l"] = []
        m.loss_registry["kl_z"] = []
        m.loss_registry["reconstruction"] = []
        m.loss_registry["total_loss"] = []
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
            end
            train_steps += 1
        end
        training_args.register_losses && register_losses!(m, Float32.(adata.countmatrix[train_inds,:]'); kl_weight=kl_weight)
    end
    @info "training complete!"
    adata.is_trained = true
    return m, adata
end

"""
    train_supervised_model!(m::scVAE, adata::AnnData, labels::AbstractVecOrMat{S}, training_args::TrainingArgs) where S <: Real

Trains a `scVAE` model on an `AnnData` object, where the latent representation is additionally trained in a supervised way to match the provided `labels`, 
where the behaviour is controlled by a `TrainingArgs` object: 

Defines the ADAM SGD optimiser, collects the model parameters, optionally splits data in training and testdata and 
initialises a `Flux.DataLoader` storing the data in the countmatrix of the `AnnData` object
and the corresponding `labels` for the supervision of the latent representation in batches. 

The loss function used is the ELBO with an additional supervised term (can be checked in the function `supervised_loss` in `src/ModelFunctions.jl`: 
In addition to the `scVAE` model and the count data, it has as additional input the provided `labels`, that need to have the same dimension as the latent represenation. 
The mean squared error between the latent representation 
and the labels is calculated and added to the standard ELBO loss. 

Updates the model parameters via stochastic gradient for the specified number of epochs, 
optionally prints out progress and current loss values. 

Returns the trained `scVAE` model.
"""
function train_supervised_model!(m::scVAE, adata::AnnData, labels::AbstractVecOrMat{S}, training_args::TrainingArgs) where S <: Real

    @assert size(labels) == (adata.ncells, m.n_latent)

    opt = Flux.Optimiser(Flux.Optimise.WeightDecay(training_args.weight_decay), ADAM(training_args.lr))
    ps = Flux.params(m)

    if training_args.train_test_split
        trainsize = training_args.trainsize
        validationsize = 1 - trainsize # nothing 
        train_inds = shuffle!(collect(1:adata.ncells))[1:Int(ceil(trainsize*adata.ncells))]
    else
        train_inds = collect(1:adata.ncells);
    end

    dataloader = Flux.DataLoader((adata.countmatrix[train_inds,:]', labels[train_inds,:]'), batchsize=training_args.batchsize, shuffle=true)

    train_steps=0
    @info "Starting training for $(training_args.max_epochs) epochs..."
    if training_args.progress
        progress = Progress(length(dataloader)*training_args.max_epochs);
    end

    for epoch in 1:training_args.max_epochs
        kl_weight = get_kl_weight(training_args.n_epochs_kl_warmup, training_args.n_steps_kl_warmup, epoch, 0)
        for d in dataloader
            curloss, back = Flux.pullback(ps) do 
                supervised_loss(m, d...; kl_weight=kl_weight)    
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
    end
    return m
end