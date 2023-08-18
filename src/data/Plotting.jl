#-------------------------------------------------------------------------------------
# filtering
#-------------------------------------------------------------------------------------
"""
    function plot_histogram(adata::AnnData, 
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
- `adata`: AnnData object
- `cell_gene`: one of `:cell` or `gene`; whether to plot counts per gene or per cell
- `counts_number`: one of `:counts` or `:number`; whether to plot counts or number of genes/cells

# Keyword arguments
- `cutoff`: cutoff to plot on top of the histogram
- `log_transform`: whether to log transform the counts
- `save_plot`: whether to save the plot to a file
- `filename`: filename to save the plot to

# Returns
- `hist`: the plot object
"""
plot_histogram(adata::AnnData, args...; kwargs...) = plot_histogram(adata.X, args...; kwargs...)

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
# highest expressed genes 
#-------------------------------------------------------------------------------------

"""
    highest_expressed_genes(adata::AnnData; 
        n_top::Int=30, 
        gene_symbols::Union{String, Nothing}=nothing,
        save_plot::Bool=false,
        filename::String="highest_expressed_genes.pdf"
    )

The function computes for each gene the fraction of counts assigned to that gene within a cell. 
The `n_top` genes with the highest mean fraction over all cells are plotted as boxplots.

# Arguments
- `adata`: AnnData object

# Keyword arguments
- `n_top`: number of genes to plot
- `gene_symbols`: column name of `adata.var` to use as gene names. If `nothing`, `adata.var_names` will be used.
- `save_plot`: whether to save the plot to a file
- `filename`: filename to save the plot to

# Returns
- the plot object
"""
highest_expressed_genes(adata::AnnData; kwargs...) = highest_expressed_genes(adata.X, kwargs...)

function highest_expressed_genes(countmatrix::AbstractMatrix;
    n_top::Int=30, 
    gene_symbols::Union{String, Nothing}=nothing,
    save_plot::Bool=false,
    filename::String="highest_expressed_genes.pdf"
    )

    if !isnothing(gene_symbols) 
        if !hasproperty(adata.var, gene_symbols) 
            @warn("If provided, `gene_symbols` must be a column name of `adata.var`. 
            Defaulting to using `adata.var_names`"
            )
            gene_symbols = adata.var_names
        else
            gene_symbols = adata.var[gene_symbols]
        end
    else
        gene_symbols = adata.var_names
    end

    # compute fraction of counts assigned to each gene
    frac_counts = countmatrix ./ sum(countmatrix, dims=1)
    mean_frac_counts = vec(mean(frac_counts, dims=2))
    gene_inds = sortperm(mean_frac_counts, rev=true)[1:n_top]
    plot_df = DataFrame(frac_counts[:,gene_inds], gene_symbols[gene_inds])
    # create boxplot 
    boxplot = stack(plot_df) |> @vlplot(:boxplot,
        x={:variable, title="gene"}, 
        y={:value, title="fraction of counts"}
    )

    save_plot && save(filename, boxplot)

    return boxplot 

end

"""
    function plot_highly_variable_genes(adata::AnnData;
        log_transform::Bool=false,
        save_plot::Bool=false,
        filename::String="highly_variable_genes.pdf"
        )
    end

Plot dispersions or normalized variance versus means for genes

# Arguments
- `adata`: AnnData object

# Keyword arguments
- `log_transform`: whether to log transform the counts
- `save_plot`: whether to save the plot to a file
- `filename`: filename to save the plot to

# Returns
- the plot object
"""
function plot_highly_variable_genes(adata::AnnData;
    log_transform::Bool=true,
    save_plot::Bool=false,
    filename::String="highly_variable_genes.pdf"
    )

    if !haspropery(adata.var, "highly_variable")
        throw(ArgumentError("No highly variable genes found. 
        Please run `highly_variable_genes!` on the `AnnData` object first."))
    end

    gene_subset = adata.var.highly_variable

    means = adata.var.means
    vars = adata.var.variances
    vars_norm = adata.var.variances_norm

    if log_transform
        means = log10.(means)
        vars = log10.(vars)
        vars_norm = log10.(vars_norm)
    end

    @vlplot() +
    [
        @vlplot(:point,
            x = {means, type = "quantitative"},
            y = {vars, type = "quantitative"},
            color = {gene_subset, type = "nominal", title = "highly variable", scale = {scheme=["#c7c7c7", "#666666"]}},
            width = 200, height = 200
        ) @vlplot(:point, 
            x = {means, type = "quantitative"},
            y = {vars_norm, type = "quantitative"}, 
            color = {gene_subset, type = "nominal", title = "highly variable"},
            width = 200, height = 200
        )
    ]

    save_plot && save(filename, plot)

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

# Keyword arguments
- `color_by`: column name of `adata.obs` to color the plot by, or a gene name to color by expression. 
    If neither is provided, celltypes will be used if present, otherwise all cells will be colored the same.
- `pcs`: which PCs to plot
- `recompute`: whether to recompute the PCA embedding or use an already existing one
- `save_plot`: whether to save the plot to a file
- `filename`: filename to save the plot to

# Returns
- the plot object
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
    elseif hasproperty(adata.obs, color_by)
        plotcolor = adata.obs[!, color_by]
    elseif color_by ∈ adata.var_names
        plotcolor = adata.layers["normalized"][:, findfirst(adata.var_names .== color_by)]
    else
        throw(ArgumentError("`color_by` argument must be a column name of either `adata.obs` or `adata.var`"))
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

# Keyword arguments
- `color_by`: column name of `adata.obs` to color the plot by, or a gene name to color by expression. 
    If neither is provided, celltypes will be used if present, otherwise all cells will be colored the same.
- `recompute`: whether to recompute the UMAP embedding  
- `save_plot`: whether to save the plot to a file
- `filename`: filename to save the plot to

# Returns
- the plot object
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
    elseif hasproperty(adata.obs, color_by)
        plotcolor = adata.obs[!, color_by]
    elseif color_by ∈ adata.var_names
        plotcolor = adata.layers["normalized"][:, findfirst(adata.var_names .== color_by)]
    else
        throw(ArgumentError("`color_by` argument must be a column name of either `adata.obs` or `adata.var`"))
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