# The scLDVAE model 

```@docs 
scLinearDecoder
``` 

```@docs
scLinearDecoder(n_input, n_output; 
    bias::Bool=true,
    dispersion::Symbol=:gene,
    gene_likelihood::Symbol=:zinb,
    dropout_rate::Float32=0.0f0,
    use_batch_norm::Bool=true,
    use_layer_norm::Bool=false
    ) 
```

```@docs 
scLDVAE(n_input::Int;
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
```