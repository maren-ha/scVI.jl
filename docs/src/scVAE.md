# The scVAE model 

## Encoder 

The implementation is based on the Python implementation of the  [`scvi-tools` encoder](https://github.com/scverse/scvi-tools/blob/b33b42a04403842591c04e414c8bb4099eaf7006/scvi/nn/_base_components.py#L202).


```@docs
scEncoder
```

```@docs
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
```

## Decoder 

The implementation is based on the Python implementation of the  [`scvi-tools` decoder](https://github.com/scverse/scvi-tools/blob/b33b42a04403842591c04e414c8bb4099eaf7006/scvi/nn/_base_components.py#L308).

There are several different distributions functions to be parameterized by the decoder, which the user can set by the `gene_likelihood` argument. The following distributions are available:
- `:zinb`: Zero-inflated negative binomial distribution
- `:nb`: Negative binomial distribution
- `:poisson`: Poisson distribution
- `:gaussian`: Gaussian distribution (for log-transformed data)
- `:bernoulli`: Bernoulli distribution (for binarized data)

Futher, there are different ways of calculating the dispersion parameter of the distribution, which can be set by the `dispersion` argument. The following options are available:
- `:gene`: the dispersion parameter is calculated separately for each gene across all cells 
- `:gene_cell`: the dispersion parameter is calculated for each gene in each cell 
- `:gene_batch`: the dispersion parameter is calculated for each gene in each expeerimental batch 

```@docs
scDecoder
```

```@docs
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
```

## Generative distribution functions 

```@docs
scVI.log_zinb_positive
```

```@docs
scVI.log_nb_positive
```

```@docs
scVI.log_poisson
```

```@docs
scVI.log_normal
```

```@docs
scVI.log_binary
```


## VAE model 

The implementation is a basic version of the [`scvi-tools` VAE object](https://github.com/scverse/scvi-tools/blob/b33b42a04403842591c04e414c8bb4099eaf7006/scvi/module/_vae.py#L22). 


```@docs
scVAE
```

```@docs
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
```