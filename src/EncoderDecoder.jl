#-------------------------------------------------------------
# Constants
#------------------------------------------------------------
scale_factor = 10000
eps = Float32(1e-8)
eta = Float32(1e-6)
#----------------------------------------------------
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
function scEncoder(
    n_input::Int, 
    n_output::Int;
    activation_fn::Function=leakyrelu, # to use in FC_layers, LeakyRelu
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
    # Just a fully connected layer 
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
    q = Encoder.encoder(x) # fully connedcted layer
    q_m = Encoder.mean_encoder(q)
    q_v = Encoder.var_activation.(Encoder.var_encoder(q)) .+ Encoder.var_eps
    latent = Encoder.z_transformation(reparameterize_gaussian(q_m, q_v))
    return q_m, q_v, latent
end
#-------------------------------------------------------------------------------------
# Encoder2
# create a structure for the mixture of expert encoder - Protein
#-------------------------------------------------------------------------------------

Base.@kwdef mutable struct MuEncoder
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

#-------------------------------------------------------------------------------------
# Initialize the Encoder2
# return MEEncoder
#-------------------------------------------------------------------------------------
Flux.@functor MuEncoder
function muEncoder( 
    n_input::Int, 
    n_output::Int;
    activation_fn::Function=leakyrelu, # to use in FC_layers
    bias::Bool=true,
    n_hidden::Int=128,
    n_layers::Int=1,
    dropout_rate::Float32=0.1f0,
    distribution::Symbol=:normal,
    use_activation::Bool=true,
    use_batch_norm::Bool=true,
    use_layer_norm::Bool=false,
    var_activation=nothing,
    var_eps::Float32=Float32(1e-6)
)
    encoder_me = FCLayers(n_input, n_hidden;
    activation_fn=activation_fn,      
    bias=bias,
    dropout_rate=dropout_rate,
    n_hidden=n_hidden,
    n_layers=n_layers,
    use_activation=use_activation,
    use_batch_norm=use_batch_norm,
    use_layer_norm=use_layer_norm)

    mean_encoder = Dense(n_hidden, n_output)
    var_encoder = Dense(n_hidden, n_output)

    transform = x -> Flux.softmax(x, dims=1)

    #if distribution == :normal # we just use normal for the protein modality ... 
    #    z_transformation = identity
    #elseif distribution == :ln 
    #    z_transformation = x -> softmax(x, dims=1)
    #else
    #    @warn "latent distribution has to be either `:normal` or `:ln`. Your choice $(distribution) is currently not supported, defaulting to `:normal`."
    #    distribution = :normal
    #    z_transformation = identity
    #end

    var_activation = isnothing(var_activation) ? exp : var_activation

    return MuEncoder(
    n_input=n_input, 
    n_output=n_output, 
    n_hidden=n_hidden, 
    n_layers=n_layers,
    distribution=distribution,
    dropout_rate=dropout_rate,
    encoder=encoder_me,
    mean_encoder=mean_encoder,
 
    var_activation=var_activation,
    var_encoder=var_encoder,
    var_eps=var_eps,
    z_transformation=transform)
end

#-------------------------------------------------------------------------------------
# MEEncoder forward function Encoder 2 
# return loc/mean, scale/variance
#-------------------------------------------------------------------------------------
function (Encoder_ME::MuEncoder)(x)
    e = Encoder_ME.encoder(x) # fully connected with relu activation
    lv = clamp.(Encoder_ME.var_encoder(e),Float64(12.0), Float64(-12.0)) # loc #restrict to avoid torch.exp() over/underflow   
    lv_size = size(lv,1) 

    # read_count to be used for normalizing r = r ./ scale_factor .* read_count# 
    # z_transformation is a softmax 
    return Encoder_ME.mean_encoder(e),  ((Encoder_ME.z_transformation(lv)) .* lv_size) .+ Encoder_ME.var_eps
end

function FCLayers(
    n_in, n_out; 
    activation_fn::Function=leakyrelu,
    bias::Bool=true,
    dropout_rate::Float32=0.2f0, 
    n_hidden::Int=128, 
    n_layers::Int=1, 
    use_batch_norm::Bool=true,
    use_layer_norm::Bool=false,
    use_activation::Bool=true,
    )

    if n_layers != 1 
        @warn "n_layers > 1 currently not supported; model initialization will default to one hidden layer only"
    end
    if use_activation
        activation_fn = leakyrelu
    else
        activation_fn : identity
    end

    batchnorm = use_batch_norm ? BatchNorm(n_out, momentum = Float32(0.01), ϵ = Float32(0.001)) : identity
    layernorm = use_layer_norm ? LayerNorm(n_out, affine=false) : identity

    fc_layers = Chain(
        Dense(n_in, n_out, bias=bias),
        batchnorm, 
        layernorm,
        x -> activation_fn.(x),
        Dropout(dropout_rate) # if dropout_rate > 0 
    )
    return fc_layers
end
#-----------------------------------------------------------
# Decoder 
#-----------------------------------------------------------
abstract type AbstractDecoder end 

Base.@kwdef mutable struct scDecoder <: AbstractDecoder
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
    activation_fn::Function=leakyrelu,
    bias::Bool=true,
    dispersion::Symbol=:gene,
    dropout_rate::Float32=0.0f0,
    modality_likelihood::Symbol=:zinb,
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

    if modality_likelihood ∈ [:nb, :zinb]
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

    px_dropout_decoder = (modality_likelihood == :zinb) ? Dense(n_hidden, n_output) : nothing

    return scDecoder(
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

function (Decoder::scDecoder)(z::AbstractVecOrMat{S}, library::AbstractVecOrMat{S}) where S <: Real
    #z = randn(10,1200)
    px = Decoder.px_decoder(z) # fully connected!
    px_scale = Decoder.px_scale_decoder(px) # dense with softmax applied
    px_dropout = apply_px_dropout_decoder(Decoder.px_dropout_decoder, Float32.(px))
    px_rate = exp.(library) .* px_scale # # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability) # torch.clamp(, max=12)
    px_r = apply_px_r_decoder(Decoder.px_r_decoder, px)
    return px_scale, px_r ,px_rate , px_dropout
end

#-----------------------------------------------------------
# Decoder 
#-----------------------------------------------------------
Base.@kwdef mutable struct MuDecoder <: AbstractDecoder
    n_input::Int
    n_output::Int
    n_hidden::Int=128
    n_layers::Int=1
    use_batch_norm::Bool=true
    use_layer_norm::Bool=false 
    px_decoder
    px_r_decoder
    px_scale_decoder
    d_transformation
    exp_fuc
end
Flux.@functor MuDecoder

function muDecoder(n_input, n_output; 
    activation_fn::Function=leakyrelu,
    bias::Bool=true,
    dispersion::Symbol=:gene,
    dropout_rate::Float32=0.0f0,
    protein_likelihood::Symbol=:nb,
    n_hidden::Int=128,
    n_layers::Int=1, 
    use_activation::Bool=true,
    use_batch_norm::Bool=true,
    use_layer_norm::Bool=false,
    exp_fuc::Function=exp)

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
    px_scale_decoder = Dense(n_hidden, n_output)
    px_r_decoder = Dense(n_hidden,n_output)


    #σ = x -> Flux.sigmoid(x, dims=1)
    σ = x -> Flux.sigmoid(x)

    return MuDecoder(n_input = n_input, n_output = n_output,
            n_layers = n_layers, 
            n_hidden = n_hidden,
            use_batch_norm = use_batch_norm,
            use_layer_norm = use_layer_norm,
            px_decoder = px_decoder,
            px_scale_decoder = px_scale_decoder,
            px_r_decoder = px_r_decoder,
            d_transformation = σ,
            exp_fuc = exp)
end

#-------------------------------------------------------------------------------------
# Decoder forward function
# forward throw the mixture of exper decoder
#-------------------------------------------------------------------------------------
function (Decoder::MuDecoder)(z)
    px = Decoder.px_decoder(z)
    log_r = Decoder.px_scale_decoder(px) # px_scale_decoder --> dense 
    log_r = clamp.(log_r, -12, 12) #scmm implementation
    r = Decoder.exp_fuc.(log_r)
    p = Decoder.px_r_decoder(px) # px_r_decoder --> dense 
    less = Float64(1.0-eps)
    p =  clamp.(Decoder.d_transformation.(p), eps, less)  #restrict to avoid probs = 0,1
    return r, p
end 

apply_px_dropout_decoder(px_dropout_decoder::Nothing, px::AbstractVecOrMat{S}) where S <: Real = nothing 
apply_px_dropout_decoder(px_dropout_decoder::Dense, px::AbstractVecOrMat{S}) where S <: Real = px_dropout_decoder(px)

apply_px_r_decoder(px_r_decoder::Nothing, px::AbstractVecOrMat{S}) where S <: Real = nothing 
apply_px_r_decoder(px_r_decoder::AbstractVecOrMat, px::AbstractVecOrMat{S}) where S <: Real = px_r_decoder
apply_px_r_decoder(px_r_decoder::Dense, px::AbstractVecOrMat{S}) where S <: Real = px_r_decoder(px)