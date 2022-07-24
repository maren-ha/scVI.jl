"""
    mutable struct TrainingArgs
"""
Base.@kwdef mutable struct TrainingArgs
    trainsize::Float32 = 0.9f0
    train_test_split::Bool=false
    batchsize::Int=128
    max_epochs::Int = 400
    lr::Float64 = 1e-3
    weight_decay::Float32 = 0.0f0
    n_steps_kl_warmup = nothing
    n_epochs_kl_warmup::Int=400
    progress::Bool=true
    verbose::Bool=false
    verbose_freq::Int=10
end

Flux.params(m::scVAE) = Flux.params(m.z_encoder, m.l_encoder, m.decoder)

"""
    train_model!(m::scVAE, adata::AnnData, training_args::TrainingArgs)
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
    end
    @info "training complete!"
    adata.is_trained = true
    return m, adata
end