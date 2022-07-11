
Base.@kwdef mutable struct scLinearDecoder <: AbstractDecoder 
    n_input::Int
    n_output::Int
    factor_regressor
    px_dropout_decoder
    px_r_decoder
    use_batch_norm::Bool=true
    use_layer_norm::Bool=false 
end

Flux.@functor scLinearDecoder

function scLinearDecoder(n_input, n_output; 
    bias::Bool=true,
    dispersion::Symbol=:gene,
    gene_likelihood::Symbol=:zinb,
    dropout_rate::Float32=0.0f0,
    use_batch_norm::Bool=true,
    use_layer_norm::Bool=false
    )

    factor_regressor = FCLayers(n_input, n_output; 
        bias=bias,
        dropout_rate=dropout_rate,
        n_layers=1,
        use_activation=false,
        use_batch_norm=use_batch_norm,
        use_layer_norm=use_layer_norm
    ) # n_hidden set to default (128) -- doesnt matter because it is one layer only anyway

    if gene_likelihood == :zinb
        px_dropout_decoder = FCLayers(n_input, n_output; 
            bias=bias,
            dropout_rate=dropout_rate,
            n_layers=1,
            use_activation=false,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm
        ) # n_hidden set to default (128) -- doesnt matter because it is one layer only anyway
    else
        px_dropout_decoder = nothing
    end

    if dispersion == :gene
        px_r_decoder = randn(Float32, n_output)
        #px_r= torch.nn.Parameter(torch.randn(n_input)) # 1200-element vector
        #px_r_ps = px_r.detach().numpy()
    elseif dispersion == :gene_cell
        px_r_decoder = Dense(n_hidden, n_output)
    else
        @warn "dispersion has to be one of `:gene` or `:gene_cell`. Your choice $(dispersion) is currently not supported, defaulting to `:gene`."
        dispersion = :gene
        px_r_decoder = randn(Float32, n_output)
    end

    return scLinearDecoder(
            n_input=n_input, 
            n_output=n_output,
            factor_regressor=factor_regressor,
            px_dropout_decoder=px_dropout_decoder,
            px_r_decoder=px_r_decoder,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
    )
end

function (LinearDecoder::scLinearDecoder)(z, library)
    #z = randn(10,1200)
    raw_px_scale = LinearDecoder.factor_regressor(z)
    px_scale = softmax(raw_px_scale, dims=1)
    px_dropout = LinearDecoder.px_dropout_decoder(z)
    px_rate = exp.(library) .* px_scale
    px_r = scLinearDecoder.px_r_decoder(px)
    return px_scale, px_r, px_rate, px_dropout
end


function get_loadings(dec::scLinearDecoder)
    if dec.use_batch_norm
        w = dec.factor_regressor[1].weight
        bn = dec.factor_regressor[2]
        b = bn.γ ./ sqrt.(bn.σ² .+ bn.ϵ)
        loadings = diagm(b)*w
    else
        loadings = dec.factor_regressor[1].weight 
    end
    return loadings 
end

function scLDVAE(n_input::Int;
    activation_fn::Function=relu, # to be used in all FC_layers instances
    bias::Symbol=:both,  # :both, :none, :encoder, :decoder; whether to use bias in linear layers of all FC instances in encoder/decoder
    dispersion::Symbol=:gene,
    dropout_rate::Float32=0.1f0,
    gene_likelihood::Symbol=:zinb,
    latent_distribution::Symbol=:normal,
    library_log_means=nothing,
    library_log_vars=nothing,
    log_variational::Bool=true,
    n_batch::Int=1,
    n_hidden::Int=128,
    n_latent::Int=10,
    n_layers::Int=1,
    use_activation::Symbol=:both, # :both, :none, :encoder, :decoder
    use_batch_norm::Symbol=:both, # :both, :none, :encoder, :decoder
    use_layer_norm::Symbol=:none, # :both, :none, :encoder, :decoder
    use_observed_lib_size::Bool=true,
    var_activation=nothing,
    var_eps::Float32=Float32(1e-4),
    seed::Int=1234
    )

    Random.seed!(seed)

    if !use_observed_lib_size
        if isnothing(library_log_means) || isnothing(library_log_vars)
            error("if not using observed library size, must provide library_log_means and library_log_vars")
        end
        # + some register_buffer thing I ignored for now: https://github.com/scverse/scvi-tools/blob/b33b42a04403842591c04e414c8bb4099eaf7006/scvi/module/_vae.py#L129
    end

    if !(gene_likelihood ∈ [:zinb, :nb, :poisson])
        @warn "gene likelihood has to be one of `:zinb`, `:nb`, or `:poisson`. Your choice $(gene_likelihood) is not supported, defaulting to `:zinb`."
    end

    use_activation_encoder = (use_activation == :encoder || use_activation == :both) # true
    (use_activation == :decoder || use_activation == :both) && @warn "Using an activation function for the decoder is not supported in a LDVAE model, choice will be overridden"
    use_activation_decoder = false

    bias_encoder = (bias == :encoder || bias == :both) # true
    bias_decoder = (bias == :decoder || bias == :both) # true

    use_batch_norm_encoder = (use_batch_norm == :encoder || use_batch_norm == :both) # true
    use_batch_norm_decoder = (use_batch_norm == :decoder || use_batch_norm == :both) # true

    use_layer_norm_encoder = (use_layer_norm == :encoder || use_layer_norm == :both) # false 

    (use_layer_norm == :decoder || use_layer_norm == :both) && @warn "Performing layer normalisation in the decoder is not supported in a LDVAE model, choice will be overridden"
    use_layer_norm_decoder = false

    # z encoder goes from the n_input-dimensional data to an n_latent-d latent space representation
    n_input_encoder = n_input

    z_encoder = scEncoder(
        n_input_encoder, 
        n_latent; 
        activation_fn=activation_fn, 
        bias=bias_encoder, 
        n_hidden=n_hidden,
        n_layers=n_layers,
        dropout_rate=dropout_rate,
        distribution=latent_distribution,
        use_activation=use_activation_encoder, 
        use_batch_norm=use_batch_norm_encoder,
        use_layer_norm=use_layer_norm_encoder,
        var_activation=var_activation,
        var_eps=var_eps
    )
    # l encoder goes from n_input-dimensional data to 1-d library size
    if use_observed_lib_size
        l_encoder = nothing
    else
        l_encoder = scEncoder(
            n_input_encoder,
            1;
            activation_fn=activation_fn, 
            bias=bias_encoder, 
            n_hidden=n_hidden,
            n_layers=1,
            dropout_rate=dropout_rate,
            use_activation=use_activation_encoder, 
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation
        )
    end
    # decoder goes from n_latent to n_input-dimensional reconstruction 
    n_input_decoder = n_latent
    decoder = scLinearDecoder(n_input_decoder, n_input;
        bias=bias_decoder, 
        dispersion=dispersion,
        gene_likelihood=gene_likelihood,
        dropout_rate=0.0f0,
        use_batch_norm=use_batch_norm_decoder,
        use_layer_norm=use_layer_norm_decoder
    )

    return scVAE(n_input=n_input,
        n_batch=n_batch, 
        n_hidden=n_hidden,
        n_latent=n_latent,
        n_layers=n_layers,
        dispersion=dispersion,
        dropout_rate=dropout_rate,
        gene_likelihood=gene_likelihood,
        latent_distribution=latent_distribution,
        log_variational=log_variational,
        use_observed_lib_size=use_observed_lib_size,
        z_encoder=z_encoder,
        l_encoder=l_encoder,
        decoder=decoder
    )
end