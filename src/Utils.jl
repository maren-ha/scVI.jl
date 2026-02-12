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
    n_hidden::Union{Int, Vector{Int}}=128, 
    n_layers::Int=1, 
    use_batch_norm::Bool=true,
    use_layer_norm::Bool=false,
    use_activation::Bool=true,
    )

    #if n_layers != 1 
    #    @warn "n_layers > 1 currently not supported; model initialization will default to one hidden layer only"
    #end

    activation_fn = use_activation ? activation_fn : identity

    l_n_hid = length(n_hidden)
    if l_n_hid > 1
        if l_n_hid != n_layers-1
            @warn "number of inner hidden layers is $(l_n_hid), but number of inner layers is $(n_layers-1), needs to coincide! Defaulting to n_hidden = 128 for all layers"
            n_hidden = 128
        end
        innerdims = [n_hidden[i] for i in 1:n_layers-1]
    else
        innerdims = [n_hidden[1] for _ in 1:n_layers-1] # "[1]" is necessary in case n_hidden is a 1-element list here (if n_layers = 2)
    end

    layerdims = [n_in, innerdims..., n_out]
    
    layerlist = [
        Chain(
            Dense(layerdims[i], layerdims[i+1], bias=bias),
            use_batch_norm ? BatchNorm(layerdims[i+1], momentum = Float32(0.01), eps = Float32(0.001)) : identity,
            use_layer_norm ? LayerNorm(layerdims[i+1], affine=false) : identity,
            x -> activation_fn.(x),
            Dropout(dropout_rate) # if dropout_rate > 0 
        ) 
        for i in 1:n_layers
    ]

    fc_layers = n_layers == 1 ? layerlist[1] : Chain(layerlist...)

    return fc_layers
end

check_layer_exists(adata, use_rep) = haskey(adata.layers, use_rep)
check_obsm_exists(adata, use_rep) = haskey(adata.obsm, use_rep)