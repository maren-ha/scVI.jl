#-------------------------------------------------------------------------------------
# Utils 
#-------------------------------------------------------------------------------------

function reparameterize_gaussian(mean, var)
    # Julia Distributions, like torch, parameterizes the Normal with std, not variance
    # Normal(μ, σ)      # Normal distribution with mean μ and variance σ^2
    return mean + sqrt.(var) .* randn(Float32, size(mean))
    #Normal(mu, var.sqrt()).rsample() # = mu + var.sqrt() * eps where eps = standard_normal(shape_of_sample)  
end

function FCLayers(
    n_in, n_out; 
    activation_fn::Function=relu,
    bias::Bool=true,
    dropout_rate::Float32=0.1f0, 
    n_hidden::Int=128, 
    n_layers::Int=1, 
    use_batch_norm::Bool=true,
    use_layer_norm::Bool=false,
    use_activation::Bool=true,
    )

    if n_layers != 1 
        @warn "n_layers > 1 currently not supported; model initialization will default to one hidden layer only"
    end

    activation_fn = use_activation ? activation_fn : identity

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
