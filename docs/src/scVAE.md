# The scVAE model 

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
