#--------------------------------------------------------------------------
# scVAE model 
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# Define and init model  
#--------------------------------------------------------------------------

"""
    mutable struct scVAE

Julia implementation of the single-cell Variational Autoencoder model corresponding to the [`scvi-tools` VAE object](https://github.com/scverse/scvi-tools/blob/b33b42a04403842591c04e414c8bb4099eaf7006/scvi/module/_vae.py#L22). 
Collects all information on the model parameters such as distribution choices and stores the model encoder and decoder. 
Can be constructed using keywords. 

**Keyword arguments**
-------------------------
 - `n_input::Ind`: input dimension = number of genes/features
 - `n_batch::Int=0`: number of batches in the data 
 - `n_hidden::Int=128`: number of hidden units to use in each hidden layer 
 - `n_latent::Int=10`: dimension of latent space 
 - `n_layers::Int=1`: number of hidden layers in encoder and decoder 
 - `dispersion::Symbol=:gene`: can be either `:gene` or `:gene-cell`. The Python `scvi-tools` options `:gene-batch` and `gene-label` are planned, but not supported yet. 
 - `is_trained::Bool=false`: indicating whether the model has been trained or not
 - `dropout_rate`: Dropout to use in the encoder and decoder layers. Setting the rate to 0.0 corresponds to no dropout. 
 - `gene_likelihood::Symbol=:zinb`: which generative distribution to parameterize in the decoder. Can be one of `:nb` (negative binomial), `:zinb` (zero-inflated negative binomial), or `:poisson` (Poisson). 
 - `latent_distribution::Symbol=:normal`: whether or not to log-transform the input data in the encoder (for numerical stability)
 -  `library_log_means::Union{Nothing, Vector{Float32}}`: log-transformed means of library size; has to be provided when not using observed library size, but encoding it
 -  `library_log_vars::Union{Nothing, Vector{Float32}}`: log-transformed variances of library size; has to be provided when not using observed library size, but encoding it
 - `log_variational`: whether or not to log-transform the input data in the encoder (for numerical stability)
 - `loss_registry::Dict=Dict()`: dictionary in which to record the values of the different loss components (reconstruction error, KL divergence(s)) during training 
 - `use_observed_lib_size::Bool=true`: whether or not to use the observed library size (if `false`, library size is calculated by a dedicated encoder)
 - `z_encoder::scEncoder`: Encoder struct of the VAE model for latent representation; see `scEncoder`
 - `l_encoder::Union{Nothing, scEncoder}`: Encoder struct of the VAE model for the library size (if `use_observed_lib_size==false`), see `scEncoder`
 - `decoder::AbstractDecoder`: Decoder struct of the VAE model; see `scDecoder`
"""
Base.@kwdef mutable struct scVAE
    n_input::Int
    n_batch::Int=0
    n_hidden::Union{Int,Vector{Int}}=128
    n_latent::Int=10
    n_layers::Int=1
    dispersion::Symbol=:gene
    dropout_rate::Float32=0.0f0
    is_trained::Bool=false
    gene_likelihood::Symbol=:zinb
    latent_distribution::Symbol=:normal
    library_log_means::Union{Nothing, Vector{Float32}}
    library_log_vars::Union{Nothing, Vector{Float32}}
    log_variational::Bool=true
    loss_registry::Dict=Dict()
    use_observed_lib_size::Bool=true
    z_encoder::scEncoder
    l_encoder::Union{Nothing, scEncoder}
    decoder::AbstractDecoder
end

"""
    scVAE(n_input::Int;
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
        n_hidden::Union{Int,Vector{Int}}=128,
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

Constructor for the `scVAE` model struct. Initialises an `scVAE` model with the parameters specified in the input arguments. 
Basic Julia implementation of the [`scvi-tools` VAE object](https://github.com/scverse/scvi-tools/blob/b33b42a04403842591c04e414c8bb4099eaf7006/scvi/module/_vae.py#L22). 

**Arguments:**
------------------------
- `n_input`: input dimension = number of genes/features

**Keyword arguments**
-------------------------
 - `activation_fn`: function to use as activation in all neural network layers of encoder and decoder 
 - `bias`: whether or not to use bias parameters in the neural network layers of encoder and decoder
 - `dispersion`: can be either `:gene` or `:gene-cell`. The Python `scvi-tools` options `:gene-batch` and `gene-label` are planned, but not supported yet. 
 - `dropout_rate`: Dropout to use in the encoder and decoder layers. Setting the rate to 0.0 corresponds to no dropout. 
 - `gene_likelihood`: which generative distribution to parameterize in the decoder. Can be one of `:nb` (negative binomial), `:zinb` (zero-inflated negative binomial), or `:poisson` (Poisson). 
 - `library_log_means`: log-transformed means of library size; has to be provided when not using observed library size, but encoding it
 - `library_log_vars`: log-transformed variances of library size; has to be provided when not using observed library size, but encoding it
 - `log_variational`: whether or not to log-transform the input data in the encoder (for numerical stability)
 - `n_batch`: number of batches in the data 
 - `n_hidden`: number of hidden units to use in each hidden layer (if an `Int` is passed, this number is used in all hidden layers, 
     alternatively an array of `Int`s can be passed, in which case the kth element corresponds to the number of units in the kth layer.
 - `n_latent`: dimension of latent space 
 - `n_layers`: number of hidden layers in encoder and decoder 
 - `use_activation`: whether or not to use an activation function in the neural network layers of encoder and decoder; if `false`, overrides choice in `actication_fn`
 - `use_batch_norm`: whether to apply batch normalization in the encoder/decoder layers; can be one of `:encoder`, `:decoder`, `both`, `:none`
 - `use_layer_norm`: whether to apply layer normalization in the encoder/decoder layers; can be one of `:encoder`, `:decoder`, `both`, `:none`
 - `use_observed_lib_size`: whether or not to use the observed library size (if `false`, library size is calculated by a dedicated encoder)
 - `var_activation`: whether or not to use an activation function for the variance layer in the encoder
 - `var_eps`: numerical stability constant to add to the variance in the reparameterisation of the latent representation
 - `seed`: random seed to use for initialization of model parameters; to ensure reproducibility. 
"""
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
    n_hidden::Union{Int,Vector{Int}}=128,
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
            throw(ArgumentError("If not using observed library size, must provide library_log_means and library_log_vars"))
        end
        # + some register_buffer thing I ignored for now: https://github.com/scverse/scvi-tools/blob/b33b42a04403842591c04e414c8bb4099eaf7006/scvi/module/_vae.py#L129
    end

    if !(gene_likelihood ∈ [:zinb, :nb, :poisson, :gaussian, :bernoulli])
        @warn "gene likelihood has to be one of `:zinb`, `:nb`, `:poisson`, `gaussian`, or `bernoulli`. Your choice $(gene_likelihood) is not supported, defaulting to `:nb`."
        gene_likelihood = :nb
    end

    l_n_hid = length(n_hidden)
    if l_n_hid > 1
        if l_n_hid != n_layers-1
            @warn "number of inner hidden layers is $(l_n_hid), but number of inner layers is $(n_layers-1), needs to coincide! Defaulting to n_hidden = 128 for all layers"
            n_hidden = 128
        end
    end

    #if (gene_likelihood == :gaussian) && log_variational
    #    @warn "log-transforming the input data is not supported for Gaussian likelihood. Setting `log_variational` to `false`."
    #    log_variational = false
    #end
    
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
    n_hidden_encoder = n_hidden
    n_hidden_decoder = length(n_hidden) > 1 ? reverse(n_hidden) : n_hidden # if n_hidden is an array, reverse it for the decoder 

    z_encoder = scEncoder(
        n_input_encoder, 
        n_latent; 
        activation_fn=activation_fn, 
        bias=bias_encoder, 
        n_hidden=n_hidden_encoder,
        n_layers=n_layers,
        distribution=latent_distribution,
        dropout_rate=dropout_rate,
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
            n_hidden=n_hidden_encoder,
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
        n_hidden=n_hidden_decoder,
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
        library_log_means=library_log_means,
        library_log_vars=library_log_vars,
        log_variational=log_variational,
        use_observed_lib_size=use_observed_lib_size,
        z_encoder=z_encoder,
        l_encoder=l_encoder,
        decoder=decoder
    )
end

function Base.summary(m::scVAE)
    training_status = m.is_trained ? "trained" : "not trained"
    println("SCVI Model with the following parameters:
     n_hidden: $(m.n_hidden), n_latent: $(m.n_latent), n_layers: $(m.n_layers),
     dropout_rate:$(m.dropout_rate),
     dispersion: $(m.dispersion),
     gene_likelihood: $(m.gene_likelihood),
     latent_distribution: $(m.latent_distribution),
     training status: $(training_status)"
    )
end