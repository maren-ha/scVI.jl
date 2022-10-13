Base.@kwdef mutable struct TrainingArgs_
    trainsize::Float32 = 0.9f0
    train_test_split::Bool=false
    batchsize::Int=64
    max_epochs::Int = 20
    lr::Float64 = 1e-4
    weight_decay::Float32 = 0.0f0
    n_steps_kl_warmup = nothing
    n_epochs_kl_warmup::Int=1
    progress::Bool=false
    verbose::Bool=true
    verbose_freq::Int=0
    log_path::String
end

Flux.params(m::scmmVAE) = Flux.params(m.encoder, m.decoder)

function train_model!(m::scmmVAE, adata::AnnData, training_args::TrainingArgs,logger)

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
                loss_(m, d; kl_weight=kl_weight)    
            end
            grad = back(1f0)
            Flux.Optimise.update!(opt, ps, grad)
            losscallback ,weighted_kl_local = loss_callback(m, d; kl_weight=kl_weight)
            if training_args.progress
                next!(progress; showvalues=[(:loss, curloss)])   
            elseif (train_steps % training_args.verbose_freq == 0) && training_args.verbose
                @info "epoch $(epoch):" loss=curloss
                with_logger(logger) do
                    @info string("Protein_Loss") weighted_kl_local=weighted_kl_local
                    @info string("training_loss") Protein_loss=losscallback
                end
            end
            train_steps += 1            
        end
    end
    @info "training complete!"
    m.is_trained = true
    return m, adata
end
##########################################################################################################################

function loss_callback(m::scmmVAE, x::AbstractMatrix{S}; kl_weight::Float32=1.0f0) where S <: Real 
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
    return lossval ,weighted_kl_local
end