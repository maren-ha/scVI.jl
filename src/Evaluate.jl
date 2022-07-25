"""
    register_latent_representation!(adata::AnnData, m::scVAE)

Calculates the latent representation obtained from encoding the `countmatrix` of the `AnnData` object 
with a trained `scVAE` model by applying the function `get_latent_representation(m, adata.countmatrix)`. 
Stored the latent representation in the `scVI_latent` field of the input `AnnData` object. 

Returns the modified `AnnData` object.
"""
function register_latent_representation!(adata::AnnData, m::scVAE)
    adata.scVI_latent = get_latent_representation(m, adata.countmatrix)
    @info "latent representation added"
    return adata 
end

"""
    register_umap_on_latent!(adata::AnnData, m::scVAE)

Calculates a UMAP (Uniform Manifold Projection and Embedding, [McInnes et al. 2018](https://arxiv.org/abs/1802.03426)) embedding of the latent representation obtained from encoding the `countmatrix` of the `AnnData` object 
with a trained `scVAE` model. If a latent representation is already stored in `adata.scVI_latent`, this is used for calculating 
the UMAP, if not, a latent representation is calculated and registered by calling `register_latent_representation!(adata, m)`. 

The UMAP is calculated using the Julia package [UMAP.jl](https://github.com/dillondaudert/UMAP.jl) with default parameters. 
It is then stored in the `scVI_latent_umap` field of the input `AnnData` object. 

Returns the modified `AnnData` object.    
"""
function register_umap_on_latent!(adata::AnnData, m::scVAE)
    if isnothing(adata.scVI_latent)
        @info "no latent representation saved in AnnData object, calculating based on scVAE model..."
        register_latent_representation!(adata, m)
    end
    adata.scVI_latent_umap = umap(adata.scVI_latent, 2; min_dist=0.3)
    @info "UMAP of latent representation added"
    return adata
end

"""
    function plot_umap_on_latent(
        m::scVAE, adata::AnnData; 
        save_plot::Bool=false, 
        seed::Int=987, 
        filename::String="UMAP_on_latent.pdf"
    )

Plots a UMAP embedding of the latent representation obtained from encoding the countmatrix of the `AnnData` object with the `scVAE` model. 
If no UMAP representation is stored in `adata.scVI_latent_umap`, it is calculated and registered by calling `register_umap_on_latent(adata, m)`.

By default, the cells are color-coded according to the `celltypes` field of the `AnnData` object. 

!TODO: add fallback for missing celltype annotation (`adata.celltypes = nothing`)

For plotting, the [VegaLite.jl](https://www.queryverse.org/VegaLite.jl/stable/) package is used.

**Arguments:**
---------------
 - `m::scVAE`: trained `scVAE` model to use for embedding the data with the model encoder
 - `adata:AnnData`: data to embed with the model; `adata.countmatrix` is encoded with `m`

 **Keyword arguments:**
 -------------------
 - `save_plot::Bool=true`: whether or not to save the plot
 - `filename::String="UMAP_on_latent.pdf`: filename under which to save the plot. Has no effect if `save_plot==false`.
 - `seed::Int=987`: which random seed to use for calculating UMAP (to ensure reproducibility)
"""
function plot_umap_on_latent(
    m::scVAE, adata::AnnData; 
    save_plot::Bool=false, 
    filename::String="UMAP_on_latent.pdf",
    seed::Int=987
    )

    if isnothing(adata.scVI_latent) 
        @info "no latent representation saved in AnnData object, calculating based on scVAE model..."
        register_latent_representation!(adata, m)
    end

    if isnothing(adata.scVI_latent_umap)
        @info "no UMAP of latent representation saved in AnnData object, calculating it now..."
        Random.seed!(seed)
        register_umap_on_latent!(adata, m)
    end

    umap_plot = @vlplot(:point, 
                        title="UMAP of scVI latent representation", 
                        x=adata.scVI_latent_umap[1,:], 
                        y = adata.scVI_latent_umap[2,:], 
                        color = adata.celltypes, 
                        width = 800, height=500
    )
    save_plot && save(filename, umap_plot)
    return umap_plot
end

function standardize(x)
    (x .- mean(x, dims = 1)) ./ std(x, dims = 1)
end

function prcomps(mat, standardizeinput = true)
    if standardizeinput
        mat = standardize(mat)
    end
    u,s,v = svd(mat)
    prcomps = u * Diagonal(s)
    return prcomps
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

!TODO: add fallback for missing celltype annotation (`adata.celltypes = nothing`)

For plotting, the [VegaLite.jl](https://www.queryverse.org/VegaLite.jl/stable/) package is used.

**Arguments:**
---------------
- `m::scVAE`: trained `scVAE` model to use for embedding the data with the model encoder
- `adata:AnnData`: data to embed with the model; `adata.countmatrix` is encoded with `m`

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

    if isnothing(adata.scVI_latent)
        @info "no latent representation saved in AnnData object, calculating based on scVAE model..."
        adata.scVI_latent = get_latent_representation(m, adata.countmatrix)
        @info "latent representation added"
    end

    pca_input = adata.scVI_latent'
    pcs = prcomps(pca_input)

    pca_plot = @vlplot(:point, 
                        title="PCA of scVI latent representation", 
                        x = pcs[:,1], 
                        y = pcs[:,2], 
                        color = adata.celltypes, 
                        width = 800, height=500
    )
    save_plot && save(filename, pca_plot)
    return pca_plot
end
