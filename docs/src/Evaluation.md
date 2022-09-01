# Model evaluation 

## Extract latent representations

```@docs
get_latent_representation
```

```@docs
register_latent_representation!
```

```@docs
get_loadings
```

## Dimension reduction and plotting

```@docs
register_umap_on_latent!
```

```@docs
plot_umap_on_latent
```

```@docs
plot_pca_on_latent
```

## Sampling from the trained model 

```@docs
sample_from_prior(m::scVAE, adata::AnnData, n_samples::Int; sample_library_size::Bool=false)
```

```@docs
sample_from_posterior(m::scVAE, adata::AnnData)
```

Both prior and posterior sampling are based on the following more low-level function, which is not exported but can be called as `scVI.decodersample`:

```@docs
decodersample(m::scVAE, z::AbstractMatrix{S}, library::AbstractMatrix{S}) where S <: Real 
```
