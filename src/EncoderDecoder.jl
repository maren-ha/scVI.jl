#-------------------------------------------------------------------------------------
# Encoder
#-------------------------------------------------------------------------------------

Base.@kwdef mutable struct scEncoder
    distribution
    dropout_rate
    encoder 
    mean_encoder 
    n_input
    n_output
    n_hidden 
    n_layers
    var_activation
    var_encoder
    var_eps
    z_transformation 
end

Flux.@functor scEncoder

# Value Types!! 
#reconstruction_loss(m::AE, x) = reconstruction_loss(m, Val(:bin), x)
#reconstruction_loss(m::AE, ::Val{:bin}, x) = -sum( x .* log.(σ.(m(x)) .+ eps(Float32)) .+ (1.0f0 .- x) .* log.(1 .- σ.(m(x)) .+ eps(Float32)))
#reconstruction_loss(m::AE, ::Val{:log}, x) = Flux.mse(m(x), x)

function scEncoder(
    n_input::Int, 
    n_output::Int;
    activation_fn::Function=relu, # to use in FC_layers
    bias::Bool=true,
    n_hidden::Int=128,
    n_layers::Int=1,
    dropout_rate::Float32=0.1f0,
    distribution::Symbol=:normal,
    use_activation::Bool=true,
    use_batch_norm::Bool=true,
    use_layer_norm::Bool=false,
    var_activation=nothing,
    var_eps::Float32=Float32(1e-4)
    )

    encoder = FCLayers(n_input, n_hidden;
        activation_fn=activation_fn,      
        bias=bias,
        dropout_rate=dropout_rate,
        n_hidden=n_hidden,
        n_layers=n_layers,
        use_activation=use_activation,
        use_batch_norm=use_batch_norm,
        use_layer_norm=use_layer_norm
    )

    mean_encoder = Dense(n_hidden, n_output)
    var_encoder = Dense(n_hidden, n_output)

    if distribution == :normal
        z_transformation = identity
    elseif distribution == :ln 
        z_transformation = x -> softmax(x, dims=1)
    else
        @warn "latent distribution has to be either `:normal` or `:ln`. Your choice $(distribution) is currently not supported, defaulting to `:normal`."
        distribution = :normal
        z_transformation = identity
    end

    var_activation = isnothing(var_activation) ? exp : var_activation

    return scEncoder(
        distribution=distribution,
        dropout_rate=dropout_rate,
        encoder=encoder,
        mean_encoder=mean_encoder,
        n_input=n_input, 
        n_output=n_output, 
        n_hidden=n_hidden, 
        n_layers=n_layers, 
        var_activation=var_activation,
        var_encoder=var_encoder,
        var_eps=var_eps,
        z_transformation=z_transformation
    )
end

function (Encoder::scEncoder)(x)
    #x = randn(n_in, batch_size)
    q = Encoder.encoder(x)
    q_m = Encoder.mean_encoder(q)
    q_v = Encoder.var_activation.(Encoder.var_encoder(q)) .+ Encoder.var_eps
    latent = Encoder.z_transformation(reparameterize_gaussian(q_m, q_v))
    return q_m, q_v, latent
end

#-------------------------------------------------------------------------------------
# supervised Encoder
#-------------------------------------------------------------------------------------

Base.@kwdef mutable struct scAEncoder
    distribution
    dropout_rate
    encoder 
    mean_encoder 
    n_input
    n_output
    n_hidden 
    n_layers
    z_transformation 
end

function scAEncoder(n_input, n_output;
    activation_fn::Function=relu,
    bias::Bool=true,
    n_hidden::Int=128, 
    n_layers::Int=1,
    dropout_rate::Float32=0.1f0,
    distribution::Symbol=:normal,
    use_activation::Bool=true,
    use_batch_norm::Bool=true,
    use_layer_norm::Bool=false)

    encoder = FCLayers(n_input, n_hidden;
        activation_fn=activation_fn,      
        bias=bias,
        dropout_rate=dropout_rate,
        n_hidden=n_hidden,
        n_layers=n_layers,
        use_activation=use_activation,
        use_batch_norm=use_batch_norm,
        use_layer_norm=use_layer_norm
    )

    mean_encoder = Dense(n_hidden, n_output)

    if distribution == :normal
        z_transformation = identity
    elseif distribution == :ln 
        z_transformation = x -> softmax(x, dims=1)
    else
        @warn "latent distribution has to be either `:normal` or `:ln`. Your choice $(distribution) is currently not supported, defaulting to `:normal`."
        distribution = :normal
        z_transformation = identity
    end

    return scAEncoder(
                distribution=distribution,
                dropout_rate=dropout_rate,
                encoder=encoder,
                mean_encoder=mean_encoder,
                n_input=n_input, 
                n_output=n_output, 
                n_hidden=n_hidden, 
                n_layers=n_layers, 
                z_transformation=z_transformation
    )
end

function (Encoder::scAEncoder)(x)
    q = Encoder.encoder(x)
    q_m = Encoder.mean_encoder(q)
    latent = Encoder.z_transformation(q_m)
    return q_m, latent
end
#-------------------------------------------------------------------------------------
# Decoder
#-------------------------------------------------------------------------------------

Base.@kwdef mutable struct scDecoder
    dispersion::Symbol=:gene
    n_input::Int
    n_output::Int
    n_hidden::Int=128
    n_layers::Int=1
    px_decoder
    px_dropout_decoder
    px_r_decoder
    px_scale_decoder
    use_batch_norm::Bool=true
    use_layer_norm::Bool=false 
end

Flux.@functor scDecoder

function scDecoder(n_input, n_output; 
    activation_fn::Function=relu,
    bias::Bool=true,
    dispersion::Symbol=:gene,
    dropout_rate::Float32=0.0f0,
    n_hidden::Int=128,
    n_layers::Int=1, 
    use_activation::Bool=true,
    use_batch_norm::Bool=true,
    use_layer_norm::Bool=false
    )

    px_decoder = FCLayers(n_input, n_hidden; 
        activation_fn=activation_fn,      
        bias=bias,
        dropout_rate=dropout_rate,
        n_hidden=n_hidden,
        n_layers=n_layers,
        use_activation=use_activation,
        use_batch_norm=use_batch_norm,
        use_layer_norm=use_layer_norm
    )

    # mean Gamma 
    px_scale_decoder = Chain(
        Dense(n_hidden, n_output), 
        x -> softmax(x, dims=1)
    )

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

    px_dropout_decoder = Dense(n_hidden, n_output)

    return scDecoder(
            dispersion=dispersion,
            n_input=n_input, 
            n_output=n_output,
            n_hidden=n_hidden,
            n_layers=n_layers, 
            px_decoder=px_decoder,
            px_dropout_decoder=px_dropout_decoder,
            px_r_decoder=px_r_decoder,
            px_scale_decoder=px_scale_decoder,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
    )
end

function (Decoder::scDecoder)(z, library)
    #z = randn(10,1200)
    px = Decoder.px_decoder(z)
    px_scale = Decoder.px_scale_decoder(px)
    px_dropout = Decoder.px_dropout_decoder(px)
    px_rate = exp.(library) .* px_scale # # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability) # torch.clamp(, max=12)
    px_r = Decoder.dispersion == :gene ? Decoder.px_r_decoder : Decoder.px_r_decoder(px)
    return px_scale, px_r, px_rate, px_dropout
end

