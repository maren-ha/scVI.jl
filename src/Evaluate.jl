"""
    register_latent_representation!(adata::AnnData, m::scVAE; name_latent::String="scVI_latent")

Calculates the latent representation obtained from encoding the `countmatrix` of the `AnnData` object 
with a trained `scVAE` model by applying the function `get_latent_representation(m, adata.X)`. 
Stored the latent representation in the `obsm` field of the input `AnnData` object as `name_latent`

Returns the modified `AnnData` object.
"""
function register_latent_representation!(adata::AnnData, m::scVAE; name_latent::String="scVI_latent")
    !m.is_trained && @warn("model has not been trained yet!")
    #adata.obsm = isnothing(adata.obsm) ? Dict() : adata.obsm
    adata.obsm[name_latent] = get_latent_representation(m, adata.X)'
    @info "latent representation added as $(name_latent)"
    return adata
end

"""
    register_umap_on_latent!(adata::AnnData, m::scVAE; name_latent::String="scVI_latent", name_umap::String="scVI_latent_umap")

Calculates a UMAP (Uniform Manifold Projection and Embedding, [McInnes et al. 2018](https://arxiv.org/abs/1802.03426)) embedding of the latent representation obtained from encoding the `countmatrix` of the `AnnData` object 
with a trained `scVAE` model. If a latent representation is already stored in `adata.name_latent`, this is used for calculating 
the UMAP, if not, a latent representation is calculated and registered by calling `register_latent_representation!(adata, m)`. 

The UMAP is calculated using the Julia package [UMAP.jl](https://github.com/dillondaudert/UMAP.jl) with default parameters. 
It is then stored in the `name_umap` field of the input `AnnData` object. 

Returns the modified `AnnData` object.    
"""
function register_umap_on_latent!(adata::AnnData, m::scVAE; name_latent::String="scVI_latent", name_umap::String="scVI_latent_umap")
    register_latent_representation!(adata, m, name_latent = name_latent)
    adata.obsm[name_umap] = umap(adata.obsm[name_latent]', 2; min_dist=0.3)'
    @info "UMAP of latent representation added as $(name_umap)"
    return adata
end

"""
    function plot_umap_on_latent(
        m::scVAE, adata::AnnData; 
        name_latent::String="scVI_latent",
        name_latent_umap::String="scVI_latent_umap",    
        save_plot::Bool=false, 
        seed::Int=987, 
        filename::String="UMAP_on_latent.pdf"
    )

Plots a UMAP embedding of the latent representation obtained from encoding the countmatrix of the `AnnData` object with the `scVAE` model. 
If no UMAP representation is stored in `adata.scVI_latent_umap`, it is calculated and registered by calling `register_umap_on_latent(adata, m)`.

By default, the cells are color-coded according to the `celltypes` field of the `AnnData` object. 

For plotting, the [VegaLite.jl](https://www.queryverse.org/VegaLite.jl/stable/) package is used.

**Arguments:**
---------------
 - `m::scVAE`: trained `scVAE` model to use for embedding the data with the model encoder
 - `adata:AnnData`: data to embed with the model; `adata.X` is encoded with `m`

 **Keyword arguments:**
 -------------------
- `name_latent::String="scVI_latent"`: name of the field in `adata.obsm` where the latent representation is stored
- `name_latent_umap::String="scVI_latent_umap"`: name of the field in `adata.obsm` where the UMAP representation is stored
 - `save_plot::Bool=true`: whether or not to save the plot
 - `filename::String="UMAP_on_latent.pdf`: filename under which to save the plot. Has no effect if `save_plot==false`.
 - `seed::Int=987`: which random seed to use for calculating UMAP (to ensure reproducibility)
"""
function plot_umap_on_latent(
    m::scVAE, adata::AnnData; 
    name_latent::String="scVI_latent",
    name_latent_umap::String="scVI_latent_umap",
    save_plot::Bool=false, 
    filename::String="UMAP_on_latent.pdf",
    seed::Int=987
    )

    plotcolor = isnothing(get_celltypes(adata)) ? fill("#ff7f0e", size(adata.X,1)) : get_celltypes(adata)

    register_latent_representation!(adata, m, name_latent=name_latent)
    register_umap_on_latent!(adata, m, name_latent=name_latent, name_umap=name_latent_umap)

    umap_plot = @vlplot(:point, 
                        title="UMAP of scVI latent representation", 
                        x = adata.obsm[name_latent_umap][:,1], 
                        y = adata.obsm[name_latent_umap][:,2], 
                        color = plotcolor,
                        width = 800, height=500
    )
    save_plot && save(filename, umap_plot)
    return umap_plot
end

"""
    function plot_latent_representation(
        m::scVAE, adata::AnnData; 
        name_latent::String="scVI_latent",
        plot_title::String="scVI latent representation",
        save_plot::Bool=false, 
        seed::Int=987, 
        filename::String="UMAP_on_latent.pdf"
    )

Plots the latent representation obtained from encoding the countmatrix of the `AnnData` object with the `scVAE` model. 
If the dimension of the latent space according to `m.n_latent` is > 2, it calculates a UMAP embedding first. 
In this case, if no UMAP representation is stored in `adata.scVI_latent_umap`, 
it is calculated and registered by calling `register_umap_on_latent(adata, m)`.

By default, the cells are color-coded according to the `celltypes` column in `adata.obs`, if present. 

For plotting, the [VegaLite.jl](https://www.queryverse.org/VegaLite.jl/stable/) package is used.

**Arguments:**
---------------
 - `m::scVAE`: trained `scVAE` model to use for embedding the data with the model encoder
 - `adata:AnnData`: data to embed with the model; `adata.X` is encoded with `m`

 **Keyword arguments:**
 -------------------
 - `name_latent::String="scVI_latent"`: name of the field in `adata.obsm` where the latent representation is stored 
 - `plot_title::String="scVI latent representation"`: title of the plot
 - `save_plot::Bool=true`: whether or not to save the plot
 - `filename::String="UMAP_on_latent.pdf`: filename under which to save the plot. Has no effect if `save_plot==false`.
 - `seed::Int=987`: which random seed to use for calculating UMAP (to ensure reproducibility)
"""
function plot_latent_representation(
    m::scVAE, adata::AnnData; 
    name_latent::String="scVI_latent",
    plot_title::String="scVI latent representation",
    save_plot::Bool=false, 
    filename::String="scvi_latent_representation.pdf",
    )

    if m.n_latent > 2 
        @info "latent space dimension is $(m.n_latent) > 2, calculating a UMAP representation... "
        latent_plot = plot_umap_on_latent(m, adata; name_latent=name_latent, save_plot=false)
    else
        plotcolor = isnothing(get_celltypes(adata)) ? fill("#ff7f0e", size(adata.X,1)) : get_celltypes(adata)
        register_latent_representation!(adata, m, name_latent=name_latent)

        latent_plot = @vlplot(:point, 
                    title=plot_title, 
                    x = adata.obsm[name_latent][:,1], 
                    y = adata.obsm[name_latent][:,2], 
                    color = plotcolor,
                    width = 800, height=500
        )    
    end
    save_plot && save(filename, latent_plot)
    return latent_plot
end

"""
    plot_pca_on_latent(
        m::scVAE, adata::AnnData; 
        save_plot::Bool=false, 
        filename::String="PCA_on_latent.pdf"
    )
    
Plots a PCA embedding of the latent representation obtained from encoding the countmatrix of the `AnnData` object with the `scVAE` model. 
If no latent representation is stored in `adata.scVI_latent`, it is calculated and registered by calling `register_latent_representation(adata, m)`.

PCA is calculated using the singular value decomposition implementation in [`LinearAlgebra.jl`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.svd), see `?LinearAlgebra.svd`. For details on the PCA implementation, see the source code in the `prcomps` function in `src/Evaluate.jl`.

By default, the cells are color-coded according to the `celltypes` field of the `AnnData` object. 

For plotting, the [VegaLite.jl](https://www.queryverse.org/VegaLite.jl/stable/) package is used.

**Arguments:**
---------------
- `m::scVAE`: trained `scVAE` model to use for embedding the data with the model encoder
- `adata:AnnData`: data to embed with the model; `adata.X` is encoded with `m`

**Keyword arguments:**
-------------------
- `save_plot::Bool=true`: whether or not to save the plot
- `filename::String="UMAP_on_latent.pdf`: filename under which to save the plot. Has no effect if `save_plot==false`.
"""
function plot_pca_on_latent(
    m::scVAE, adata::AnnData; 
    save_plot::Bool=false, 
    filename::String="PCA_on_latent.pdf"
    )

    plotcolor = isnothing(get_celltypes(adata)) ? fill("#ff7f0e", size(adata.X,1)) : get_celltypes(adata)
    register_latent_representation!(adata, m)


    pca_input = adata.obsm["scVI_latent"]
    pcs = prcomps(pca_input)

    pca_plot = @vlplot(:point, 
                        title="PCA of scVI latent representation", 
                        x = pcs[:,1], 
                        y = pcs[:,2], 
                        color = plotcolor, 
                        width = 800, height=500
    )
    save_plot && save(filename, pca_plot)
    return pca_plot
end

#
# sampling from the model 
#

"""
    decodersample(m::scVAE, z::AbstractMatrix{S}, library::AbstractMatrix{S}) where S <: Real 

Samples from the generative distribution defined by the decoder of the `scVAE` model based on values of the latent variable `z`. 
Depending on whether `z` is sampled from the prior or posterior, the function can be used to realise both prior and posterior sampling, see
`sample_from_posterior()` and `sample_from_prior` for details. 

The distribution ((zero-inflated) negative binomial or Poisson) is parametrised by `mu`, `theta` and `zi` (logits of dropout parameter). 
The implementation is adapted from the corresponding [`scvi tools` function](https://github.com/YosefLab/scvi-tools/blob/f0a3ba6e11053069fd1857d2381083e5492fa8b8/scvi/distributions/_negative_binomial.py#L420)

**Arguments:** 
-----------------
 - `m::scVAE`: `scVAE` model from which the decoder is used for sampling
 - `z::AbstractMatrix`: values of the latent representation to use as input for the decoder 
 - `library::AbstractMatrix`: library size values that are used for scaling in the decoder (either corresponding to the observed or the model-encoded library size) 
"""
function decodersample(m::scVAE, z::AbstractMatrix{S}, library::AbstractMatrix{S}) where S <: Real 
    px_scale, theta, mu, zi_logits = generative(m, z, library)
    if m.gene_likelihood == :nb
        return rand.(NegativeBinomial.(theta, theta ./ (theta .+ mu .+ eps(Float32))))
    elseif m.gene_likelihood == :zinb
        samp = rand.(NegativeBinomial.(theta, theta ./ (theta .+ mu .+ eps(Float32))))
        zi_probs = logits_to_probs(zi_logits)
        is_zero = rand(Float32, size(mu)) .<= zi_probs
        samp[is_zero] .= 0.0
        return samp
    elseif m.gene_likelihood == :poisson
        return rand.(Poisson.(mu))
    else 
        throw(ArgumentError("Not implemented"))
    end
end

"""
    sample_from_posterior(m::scVAE, adata::AnnData)

Samples from the posterior distribution of the latent representation of a trained `scVAE` model. 
Calculates the latent posterior mean and variance and the library size based on the `countmatrix` of the input `AnnData` object and samples from the posterior. 
Subsequently samples from the generative distribution defined by the decoder based on the samples of the latent representation and the library size. 

Returns the samples from the model. 

**Arguments:**
--------------
- `m::scVAE`: trained `scVAE` model from which to sample
- `adata::AnnData`: `AnnData` object based on which to calculate the latent posterior
"""
function sample_from_posterior(m::scVAE, adata::AnnData)
    sample_from_posterior(m, adata.X')
end

function sample_from_posterior(m::scVAE, x::AbstractMatrix{S}) where S <: Real 
    !m.is_trained && @warn("model has not been trained yet!")
    z, qz_m, qz_v, ql_m, ql_v, library = inference(m, x)
    return decodersample(m, z, library)
end

"""
    sample_from_prior(m::scVAE, adata::AnnData, n_samples::Int; sample_library_size::Bool=false)

Samples from the prior N(0,1) distribution of the latent representation of a trained `scVAE` model. 
Calculates the library size based on the `countmatrix` of the input `AnnData` object and either samples from it or uses the mean.
Subsequently draws `n_samples` from the generative distribution defined by the decoder based on the samples from the prior and the library size.

Returns the samples from the model. 

**Arguments:**
--------------
- `m::scVAE`: trained `scVAE` model from which to sample
- `adata::AnnData`: `AnnData` object based on which to calculate the library size
- `n_samples::Int`: number of samples to draw

**Keyword arguments:**
- `sample_library_size::Bool=false`: whether or not to sample from the library size. If `false`, the mean of the observed library size is used. 
"""
function sample_from_prior(m::scVAE, adata::AnnData, n_samples::Int; sample_library_size::Bool=false)
    sample_from_prior(m, adata.X', n_samples, sample_library_size=sample_library_size)
end

function sample_from_prior(m::scVAE, x::AbstractMatrix{S}, n_samples::Int; sample_library_size::Bool=false) where S <: Real 
    !m.is_trained && @warn("model has not been trained yet!")
    encoder_input = m.log_variational ? log.(one(S) .+ x) : x
    _, _, orig_library = get_library(m, x, encoder_input)
    # library = sample_library_size ? rand(orig_library, n_samples) : mean(orig_library)
    library = sample_library_size ? rand(Normal(mean(orig_library), std(orig_library)), n_samples) : fill(mean(orig_library), n_samples)
    z = rand(Normal(0.0f0,1.0f0), m.n_latent, n_samples)
    return decodersample(m, z, Matrix(library'))
end

#=
# to test: 
Random.seed!(42)
x = first(dataloader)
z, qz_m, qz_v, ql_m, ql_v, library = inference(m,x)
samp = decodersample(m, z, library)
=#
