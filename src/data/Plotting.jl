#-------------------------------------------------------------------------------------
# dimension reduction
#-------------------------------------------------------------------------------------

"""
    function plot_pca(
        adata::AnnData;
        color_by::String="",
        pcs::Vector{Int}=[1,2],
        recompute::Bool=false,
        save_plot::Bool=false,
        filename::String="PCA.pdf"
    )

Plot a PCA embedding on a given `AnnData` object.

# Arguments
- `adata`: AnnData object
- `color_by`: column name of `adata.obs` to color the plot by
- `pcs`: which PCs to plot
- `recompute`: whether to recompute the PCA embedding or use an already existing one
- `save_plot`: whether to save the plot to a file
- `filename`: filename to save the plot to

# Returns
- `pca_plot`: the plot object
"""
function plot_pca(
    adata::AnnData;
    color_by::String="",
    pcs::Vector{Int}=[1,2],
    recompute::Bool=false,
    save_plot::Bool=false,
    filename::String="PCA.pdf"
    )

    # check if PCA is already computed
    if !haskey(adata.obsm, "PCA") || recompute
        pca!(adata)
    end

    # set color_by argument 
    if color_by == "" 
        # try get celltypes 
        plotcolor = isnothing(get_celltypes(adata)) ? fill("#ff7f0e", size(adata.X,1)) : get_celltypes(adata)
    else
        plotcolor = adata.obs[!, color_by]
    end

    # plot
    pca_plot = @vlplot(:point, 
                        title="PCA representation", 
                        x = adata.obsm["PCA"][:,pcs[1]],
                        y = adata.obsm["PCA"][:,pcs[2]],
                        color = plotcolor,
                        width = 800, height=500
    )
    save_plot && save(filename, pca_plot)
    return pca_plot
end
 
"""
    function plot_umap(
        adata::AnnData;
        color_by::String="",
        recompute::Bool=false,
        save_plot::Bool=false,
        filename::String="UMAP.pdf"
        )

Plot a UMAP embedding on a given `AnnData` object.

# Arguments
- `adata`: AnnData object
- `color_by`: column name of `adata.obs` to color the plot by
- `recompute`: whether to recompute the UMAP embedding  
- `save_plot`: whether to save the plot to a file
- `filename`: filename to save the plot to

# Returns
- `umap_plot`: the plot object
"""
function plot_umap(
    adata::AnnData;
    color_by::String="",
    recompute::Bool=false,
    save_plot::Bool=false,
    filename::String="UMAP.pdf"
    )

    # check if UMAP is already computed
    if !haskey(adata.obsm, "umap") || recompute
        umap!(adata)
    end

    # set color_by argument
    if color_by == "" 
        # try get celltypes 
        plotcolor = isnothing(get_celltypes(adata)) ? fill("#ff7f0e", size(adata.X,1)) : get_celltypes(adata)
    else
        plotcolor = adata.obs[!, color_by]
    end

    # plot
    umap_plot = @vlplot(:point, 
                        title="UMAP representation", 
                        x = adata.obsm["umap"][:,1],
                        y = adata.obsm["umap"][:,2],
                        color = plotcolor,
                        width = 800, height=500
    )
    save_plot && save(filename, umap_plot)
    return umap_plot
end