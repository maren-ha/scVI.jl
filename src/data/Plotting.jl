#-------------------------------------------------------------------------------------
# filtering
#-------------------------------------------------------------------------------------
plot_histogram(adata::AnnData, args...; kwargs...) = plot_histogram(adata.X, args...; kwargs...)

"""
    function plot_histogram(countmatrix::AbstractMatrix, 
        cell_gene::Symbol = :gene, 
        counts_number::Symbol = :counts; 
        cutoff::Union{Nothing, Real}=nothing,
        log_transform::Bool=false, 
        save_plot::Bool=false, 
        filename::String="counts_per_gene_histogram.pdf"
        )

Plot a histogram of counts per gene or cells, 
or alternatively he number of genes expressed per cell or 
the number of cells per gene in which it is expressed.

Additionally, a user-specified cutoff can be plotted on top, 
to be used as a visualization tool for filtering.

The function is called internally when using the plotting options 
in the `filter_cells` and `filter_genes` functions.

# Arguments
-------------------
- `countmatrix`: matrix of counts
- `cell_gene`: one of `:cell` or `gene`; whether to plot counts per gene or per cell
- `counts_number`: one of `:counts` or `:number`; whether to plot counts or number of genes/cells

# Keyword arguments
------------------- 
- `cutoff`: cutoff to plot on top of the histogram
- `log_transform`: whether to log transform the counts
- `save_plot`: whether to save the plot to a file
- `filename`: filename to save the plot to

# Returns
-------------------
- `hist`: the plot object
"""
function plot_histogram(countmatrix::AbstractMatrix, 
    cell_gene::Symbol = :gene, 
    counts_number::Symbol = :counts; 
    cutoff::Union{Nothing, Real}=nothing,
    log_transform::Bool=false, 
    save_plot::Bool=false, 
    filename::String="counts_per_gene_histogram.pdf"
    )

    if cell_gene == :gene
        dims = 1
        gene_cell = :cell
    elseif cell_gene == :cell
        dims = 2
        gene_cell = :gene
    else
        throw(ArgumentError("second argument must be either :gene or :cell"))
    end

    if counts_number == :counts
        vals_to_plot = vec(sum(countmatrix, dims=dims))
        if log_transform 
            vals_to_plot = log.(vals_to_plot)
            xtitle = "log counts per $(string(cell_gene))"
        else
            vals_to_plot = counts_per_gene
            xtitle = "counts per $(string(cell_gene))"
        end
    elseif counts_number == :number
        vals_to_plot = vec(sum(countmatrix .> 0, dims=dims))
        xtitle = "number of $(string(gene_cell))s per $(string(cell_gene))"
    else
        throw(ArgumentError("third argument must be either :counts or :number"))
    end

    if isnothing(cutoff)    
        hist = @vlplot(:bar, 
            x = {vals_to_plot, bin=true, title = xtitle}, 
            y = {"count()", title = "number of $(string(cell_gene))s"}
        ) 
    else
        hist = 
        @vlplot() +
        @vlplot(:bar, 
            x = {vals_to_plot, bin=true, title = xtitle}, 
            y = {"count()", title = "number of $(string(cell_gene))s"}
        ) +
        @vlplot({:rule, 
            color="red", size=3}, 
            x = {[cutoff], type="quantitative", title = xtitle}
        )
    end

    save_plot && save(filename, hist)
    return hist
end

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