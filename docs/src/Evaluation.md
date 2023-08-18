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

```@docs
plot_latent_representation
```

## Sampling from the trained model 

```@docs
sample_from_prior
```

```@docs
sample_from_posterior
```

Both prior and posterior sampling are based on the following more low-level function, which is not exported but can be called as `scVI.decodersample`:

```@docs
scVI.decodersample
```
