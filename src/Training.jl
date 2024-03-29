"""
    mutable struct TrainingArgs

Struct to store hyperparameters to control and customise the training process of an `scVAE` model. 
Can be constructed using keywords. 

# Fields for construction
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

function setup_batch_indices_for_library_scaling(m::scVAE, adata::AnnData, batch_key::Symbol; verbose::Bool=true)

    batch_indices = ones(Int64, size(adata.X,1))
    
    l_library_means = isnothing(m.library_log_means) ? nothing : length(m.library_log_means)
    l_unique_batch_keys = hasproperty(adata.obs, batch_key) ? length(unique(adata.obs[!,batch_key])) : nothing

    if !m.use_observed_lib_size 
        # breaking conditions: if something is really wrong, default to using observed library size 
        if (isnothing(m.library_log_means) || isnothing(m.library_log_vars))
            @warn "If encoding library during training, `library_log_means` and `library_log_vars` must be provided, defaulting to using observed library size!"
            m.use_observed_lib_size=true
        elseif !hasproperty(adata.obs, batch_key)
            @warn "batch_key $(batch_key) not found in adata.obs, defaulting to using observed library size!"
            m.use_observed_lib_size=true
        elseif l_library_means != l_unique_batch_keys
            @warn "length of $(m.library_log_means) = $(l_library_means) does not match number of unique values in adata.obs[!,batch_key] = $(l_unique_batch_keys), defaulting to using observed library size!"
            m.use_observed_lib_size=true
        else 
            verbose && @info "library size will be encoded"
            # some last checks 
            if m.n_batch != l_library_means
                @warn "m.n_batch = $(m.n_batch) different from length(library_log_means) = $(l_library_means) -- overriding m.n_batch and setting to $(l_library_means)"
                m.n_batch = l_library_means
            end
            @assert m.n_batch == l_library_means == l_unique_batch_keys
            for (id_ind, batch_id) in enumerate(unique(adata.obs[!,batch_key]))
                cur_batch_inds = findall(x -> x == batch_id, adata.obs[!,batch_key])
                batch_indices[cur_batch_inds] .= id_ind
            end
        end
    elseif m.use_observed_lib_size
        if m.n_batch > 1 || (!isnothing(m.library_log_means) && l_library_means > 1)
            @warn "either m.n_batch > 1 or length of observed library_log_means/vars vector > 1, but observed library size is used, thus ignoring potential batch effects"
        end
        verbose && @info "Using observed library size in each training batch, thus ignoring potential experimental batch effects"
       @assert batch_indices == ones(Int64, size(adata.X,1))
    end

    return batch_indices
end

"""
    train_model!(m::scVAE, adata::AnnData, training_args::TrainingArgs; 
    batch_key::Symbol=:batch, layer::Union{String, Nothing}=nothing)

Trains an `scVAE` model on an `AnnData` object, where the behaviour is controlled by a `TrainingArgs` object: 
Defines the ADAM SGD optimiser, collects the model parameters, optionally splits data in training and testdata and 
initialises a `Flux.DataLoader` storing the data in the countmatrix of the `AnnData` object in batches. 
Updates the model parameters via stochastic gradient for the specified number of epochs, 
optionally prints out progress and current loss values. 

# Arguments
- `m::scVAE`: the model to train
- `adata::AnnData`: the data on which to train the model
- `training_args::TrainingArgs`: the training arguments controlling the training behaviour

# Keyword arguments
- `batch_key::Symbol=:batch`: the key in `adata.obs` on which to split the data in batches for the library encoder. 
    If `m.use_observed_lib_size==true`, this argument is ignored.
- layer::Union{String, Nothing}=nothing`: the layer in `adata.layers` on which to train the model. 
    If `m.gene_likelihood ∈ [:gaussian, :bernoulli]`, this argument is mandatory.

# Returns 
- the trained `scVAE` model.
"""
function train_model!(m::scVAE, adata::AnnData, training_args::TrainingArgs; 
    batch_key::Symbol=:batch, layer::Union{String, Nothing}=nothing)

    opt = Flux.Optimiser(Flux.Optimise.WeightDecay(training_args.weight_decay), ADAM(training_args.lr))
    ps = Flux.params(m)

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
    end

    batch_indices = setup_batch_indices_for_library_scaling(m, adata, batch_key, verbose=training_args.verbose)
    dataloader = Flux.DataLoader((X[train_inds,:]', batch_indices[train_inds]), batchsize=training_args.batchsize, shuffle=true)
    # dataloader = Flux.DataLoader(adata.X[train_inds,:]', batchsize=training_args.batchsize, shuffle=true)

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
        training_args.register_losses && register_losses!(m, Float32.(X[train_inds,:]'); kl_weight=kl_weight)
    end
    @info "training complete!"
    m.is_trained = true
    #adata.is_trained = true
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

# Arguments
- `m::scVAE`: the model to train
- `adata::AnnData`: the data on which to train the model
- `labels::AbstractVecOrMat{S}`: the labels to use for the supervised training of the latent representation
- `training_args::TrainingArgs`: the training arguments controlling the training behaviour

# Returns
- the trained `scVAE` model
"""
function train_supervised_model!(m::scVAE, adata::AnnData, labels::AbstractVecOrMat{S}, training_args::TrainingArgs) where S <: Real

    ncells, ngenes = size(adata.X)

    @assert size(labels) == (ncells, m.n_latent)

    opt = Flux.Optimiser(Flux.Optimise.WeightDecay(training_args.weight_decay), ADAM(training_args.lr))
    ps = Flux.params(m)

    if training_args.train_test_split
        trainsize = training_args.trainsize
        validationsize = 1 - trainsize # nothing 
        train_inds = shuffle!(collect(1:ncells))[1:Int(ceil(trainsize*ncells))]
    else
        train_inds = collect(1:ncells);
    end

    dataloader = Flux.DataLoader((adata.X[train_inds,:]', labels[train_inds,:]'), batchsize=training_args.batchsize, shuffle=true)

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
    @info "training complete!"
    m.is_trained = true
    #adata.is_trained = true
    return m, adata
end