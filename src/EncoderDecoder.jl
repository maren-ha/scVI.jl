#-------------------------------------------------------------------------------------
# Encoder
#-------------------------------------------------------------------------------------

"""
    mutable struct scEncoder

Julia implementation of the Encoder of the single-cell VAE model from [`scvi-tools`](https://github.com/scverse/scvi-tools/blob/b33b42a04403842591c04e414c8bb4099eaf7006/scvi/nn/_base_components.py#L202)
Collects all information on the encoder parameters and stores the basic encoder and mean and variance encoders. 
Can be constructed using keywords. 

**Keyword arguments**
-------------------------
 - `encoder`: `Flux.Chain` of fully connected layers realising the first part of the encoder (before the split in mean and variance). For details, see the source code of `FC_layers` in `src/Utils`.
 - `mean_encoder`: `Flux.Dense` fully connected layer realising the latent mean encoder 
 - `n_input`: input dimension = number of genes/features
 - `n_hidden`: number of hidden units to use in each hidden layer 
 - `n_output`: output dimension of the encoder = dimension of latent space 
 - `n_layers`: number of hidden layers in encoder and decoder 
 - `var_activation`: whether or not to use an activation function for the variance layer in the encoder
 - `var_encoder`: `Flux.Dense` fully connected layer realising the latent variance encoder 
 - `var_eps`: numerical stability constant to add to the variance in the reparameterisation of the latent representation
 - `z_transformation`: whether to apply a `softmax` transformation the latent z if assuming a lognormal instead of a normal distribution
"""
Base.@kwdef mutable struct scEncoder
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
"""
    scEncoder(
        n_input::Int, 
        n_output::Int;
        activation_fn::Function=relu, # to use in FC_layers
        bias::Bool=true,
        n_hidden::Int=128,
        n_layers::Int=1,
        distribution::Symbol=:normal,
        dropout_rate::Float32=0.1f0,
        use_activation::Bool=true,
        use_batch_norm::Bool=true,
        use_layer_norm::Bool=false,
        var_activation=nothing,
        var_eps::Float32=Float32(1e-4)
    )

Constructor for an `scVAE` encoder. Initialises an `scEncoder` object according to the input parameters. 
Julia implementation of the [scvi-tools encoder](https://github.com/scverse/scvi-tools/blob/b33b42a04403842591c04e414c8bb4099eaf7006/scvi/nn/_base_components.py#L202).

**Arguments:**
---------------------------
- `n_input`: input dimension = number of genes/features
- `n_output`: output dimension of the encoder = latent space dimension

**Keyword arguments:**
---------------------------
- `activation_fn`: function to use as activation in all encoder neural network layers 
- `bias`: whether or not to use bias parameters in the encoder neural network layers
- `n_hidden`: number of hidden units to use in each hidden layer 
- `n_layers`: number of hidden layers in encoder 
- `distribution` :whether to use a `:normal` or lognormal (`:ln`) distribution for the latent z  
- `dropout_rate`: dropout to use in all encoder layers. Setting the rate to 0.0 corresponds to no dropout. 
- `use_activation`: whether or not to use an activation function in the encoder neural network layers; if `false`, overrides choice in `actication_fn`
- `use_batch_norm`: whether or not to apply batch normalization in the encoder layers
- `use_layer_norm`: whether or not to apply layer normalization in the encoder layers
- `var_activation`: whether or not to use an activation function for the variance layer in the encoder
- `var_eps`: numerical stability constant to add to the variance in the reparameterisation of the latent representation
"""
function scEncoder(
    n_input::Int, 
    n_output::Int;
    activation_fn::Function=relu, # to use in FC_layers
    bias::Bool=true,
    n_hidden::Int=128,
    n_layers::Int=1,
    distribution::Symbol=:normal,
    dropout_rate::Float32=0.1f0,
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
    dropout_rate::Float32=0.1f0,
    distribution::Symbol=:normal,
    n_hidden::Int=128, 
    n_layers::Int=1,
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
abstract type AbstractDecoder end 

"""
    mutable struct scDecoder <: AbstractDecoder

Julia implementation of the Decoder of single-cell VAE model from [`scvi-tools`](https://github.com/scverse/scvi-tools/blob/b33b42a04403842591c04e414c8bb4099eaf7006/scvi/nn/_base_components.py#L308)
Collects all information on the decoder parameters and stores the decoder parts. 
Can be constructed using keywords. 

**Keyword arguments**
-------------------------
 - `n_input`: input dimension = dimension of latent space 
 - `n_hidden`: number of hidden units to use in each hidden layer 
 - `n_output`: output dimension of the decoder = number of genes/features
 - `n_layers`: number of hidden layers in decoder 
 - `px_decoder`: `Flux.Chain` of fully connected layers realising the first part of the decoder (before the split in mean, dispersion and dropout decoder). For details, see the source code of `FC_layers` in `src/Utils`.
 - `px_dropout_decoder`: if the generative distribution is zero-inflated negative binomial (`gene_likelihood = :zinb` in the `scVAE` model construction): `Flux.Dense` layer, else `nothing`.
 - `px_r_decoder`: decoder for the dispersion parameter. If generative distribution is not some (zero-inflated) negative binomial, it is `nothing`. Else, it is a parameter vector  or a `Flux.Dense`, depending on whether the dispersion is estimated per gene (`dispersion = :gene`), or per gene and cell (`dispersion = :gene_cell`)  
 - `px_scale_decoder`: decoder for the mean of the reconstruction, `Flux.Chain` of a `Dense` layer followed by `softmax` activation
 - `use_batch_norm`: whether or not to apply batch normalization in the decoder layers
 - `use_layer_norm`: whether or not to apply layer normalization in the decoder layers 
"""
Base.@kwdef mutable struct scDecoder <: AbstractDecoder
    n_input::Int
    n_hidden::Int=128
    n_output::Int
    n_layers::Int=1
    px_decoder
    px_dropout_decoder
    px_r_decoder
    px_scale_decoder
    use_batch_norm::Bool=true
    use_layer_norm::Bool=false 
end

Flux.@functor scDecoder

"""
    scDecoder(n_input, n_output; 
        activation_fn::Function=relu,
        bias::Bool=true,
        dispersion::Symbol=:gene,
        dropout_rate::Float32=0.0f0,
        gene_likelihood::Symbol=:zinb,
        n_hidden::Int=128,
        n_layers::Int=1, 
        use_activation::Bool=true,
        use_batch_norm::Bool=true,
        use_layer_norm::Bool=false
    )

Constructor for an `scVAE` decoder. Initialises an `scDecoder` object according to the input parameters. 
Julia implementation of the [scvi-tools decoder](https://github.com/scverse/scvi-tools/blob/b33b42a04403842591c04e414c8bb4099eaf7006/scvi/nn/_base_components.py#L308).

**Arguments:**
---------------------------
- `n_input`: input dimension of the decoder = latent space dimension
- `n_output`: output dimension = number of genes/features in the data 

**Keyword arguments:**
---------------------------
- `activation_fn`: function to use as activation in all decoder neural network layers 
- `bias`: whether or not to use bias parameters in the decoder neural network layers
- `dispersion`: whether to estimate the dispersion parameter for the (zero-inflated) negative binomial generative distribution per gene (`:gene`) or per gene and cell (`:gene_cell`) 
- `dropout_rate`: dropout to use in all decoder layers. Setting the rate to 0.0 corresponds to no dropout. 
- `n_hidden`: number of hidden units to use in each hidden layer 
- `n_layers`: number of hidden layers in decoder 
- `use_activation`: whether or not to use an activation function in the decoder neural network layers; if `false`, overrides choice in `actication_fn`
- `use_batch_norm`: whether or not to apply batch normalization in the decoder layers
- `use_layer_norm`: whether or not to apply layer normalization in the decoder layers
"""
function scDecoder(n_input, n_output; 
    activation_fn::Function=relu,
    bias::Bool=true,
    dispersion::Symbol=:gene,
    dropout_rate::Float32=0.0f0,
    gene_likelihood::Symbol=:zinb,
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

    if gene_likelihood ∈ [:nb, :zinb]
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
    else
        px_r_decoder = nothing 
    end

    px_dropout_decoder = (gene_likelihood == :zinb) ? Dense(n_hidden, n_output) : nothing

    return scDecoder(
            n_input=n_input, 
            n_hidden=n_hidden,
            n_output=n_output,
            n_layers=n_layers, 
            px_decoder=px_decoder,
            px_dropout_decoder=px_dropout_decoder,
            px_r_decoder=px_r_decoder,
            px_scale_decoder=px_scale_decoder,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
    )
end

function (Decoder::scDecoder)(z::AbstractVecOrMat{S}, library::AbstractVecOrMat{S}) where S <: Real
    #z = randn(10,1200)
    px = Decoder.px_decoder(z)
    px_scale = Decoder.px_scale_decoder(px)
    px_dropout = apply_px_dropout_decoder(Decoder.px_dropout_decoder, px)
    px_rate = exp.(library) .* px_scale # # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability) # torch.clamp(, max=12)
    px_r = apply_px_r_decoder(Decoder.px_r_decoder, px)
    return px_scale, px_r, px_rate, px_dropout
end

apply_px_dropout_decoder(px_dropout_decoder::Nothing, px::AbstractVecOrMat{S}) where S <: Real = nothing 
apply_px_dropout_decoder(px_dropout_decoder::Dense, px::AbstractVecOrMat{S}) where S <: Real = px_dropout_decoder(px)

apply_px_r_decoder(px_r_decoder::Nothing, px::AbstractVecOrMat{S}) where S <: Real = nothing 
apply_px_r_decoder(px_r_decoder::AbstractVecOrMat, px::AbstractVecOrMat{S}) where S <: Real = px_r_decoder
apply_px_r_decoder(px_r_decoder::Dense, px::AbstractVecOrMat{S}) where S <: Real = px_r_decoder(px)

#= 
# previous version based on making types callable 
(n::Nothing)(x) = nothing 
(v::Vector{Float32})(x) = v

function (Decoder::scDecoder)(z::AbstractVecOrMat{S}, library::AbstractVecOrMat{S}) where S <: Real
    #z = randn(10,1200)
    px = Decoder.px_decoder(z)
    px_scale = Decoder.px_scale_decoder(px)
    px_dropout = Decoder.px_dropout_decoder(px)
    px_rate = exp.(library) .* px_scale # # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability) # torch.clamp(, max=12)
    px_r = Decoder.px_r_decoder(px)
    return px_scale, px_r, px_rate, px_dropout
end
=# 