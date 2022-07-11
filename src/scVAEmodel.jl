#--------------------------------------------------------------------------
# scVAE model 
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# Define and init model  
#--------------------------------------------------------------------------

Base.@kwdef mutable struct scVAE
    n_input::Int
    n_batch::Int=0
    n_hidden::Int=128
    n_latent::Int=10
    n_layers::Int=1
    dispersion::Symbol=:gene
    dropout_rate::Float32=0.0f0
    gene_likelihood::Symbol=:zinb
    latent_distribution::Symbol=:normal
    log_variational::Bool=true
    use_observed_lib_size::Bool=true
    z_encoder::scEncoder
    l_encoder::Union{Nothing, scEncoder}
    decoder::AbstractDecoder
end

function scVAE(n_input::Int;
    activation_fn::Function=relu, # to be used in all FC_layers instances
    bias::Symbol=:both, # whether to use bias in all linear layers of all FC instances 
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
    use_activation::Symbol=:both, 
    use_batch_norm::Symbol=:both,
    use_layer_norm::Symbol=:none,
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
    use_activation_decoder = (use_activation == :decoder || use_activation == :both) # true

    bias_encoder = (bias == :encoder || bias == :both) # true
    bias_decoder = (bias == :decoder || bias == :both) # true

    use_batch_norm_encoder = (use_batch_norm == :encoder || use_batch_norm == :both) # true
    use_batch_norm_decoder = (use_batch_norm == :decoder || use_batch_norm == :both) # true

    use_layer_norm_encoder = (use_layer_norm == :encoder || use_layer_norm == :both) # false 
    use_layer_norm_decoder = (use_layer_norm == :decoder || use_layer_norm == :both) # false

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
    decoder = scDecoder(n_input_decoder, n_input;
        activation_fn=activation_fn, 
        bias=bias_decoder, 
        dispersion=dispersion, 
        dropout_rate=dropout_rate,
        gene_likelihood=gene_likelihood,
        n_hidden=n_hidden,
        n_layers=n_layers,
        use_activation=use_activation_decoder,
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

function Base.summary(m::scVAE)
    string("SCVI Model with the following parameters: 
     n_hidden: $(m.n_hidden), n_latent: $(m.n_latent), n_layers: $(m.n_layers), 
     dropout_rate:$(m.dropout_rate), 
     dispersion: $(m.dispersion), 
     gene_likelihood: $(m.gene_likelihood), 
     latent_distribution: $(m.latent_distribution)"
    )
end