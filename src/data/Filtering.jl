"""
    function filter_cells!(adata::AnnData; 
        min_counts::Union{Int, Nothing}=nothing, 
        min_genes::Union{Int, Nothing}=nothing, 
        max_counts::Union{Int, Nothing}=nothing, 
        max_genes::Union{Int, Nothing}=nothing,
        verbose::Bool = true, 
        plot_before::Bool=false,
        log_transform_plot::Bool=false,
        plot_after::Bool=false,
        save_plot_before::Bool=false, 
        save_plot_after::Bool=false,
        plot_filename_before::String="cell_histogram_before_filtering.pdf",
        plot_filename_after::String="cell_histogram_after_filtering.pdf"
        )
    
Filter cell outliers based on counts and numbers of genes expressed.
For instance, only keep cells with at least `min_counts` counts or
`min_genes` genes expressed. This is to filter measurement outliers,
i.e. “unreliable” observations.
Only provide one of the optional parameters `min_counts`, `min_genes`,
`max_counts`, `max_genes` per call.

# Arguments
- `adata`: `AnnData` object of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes.
- `min_counts`: Minimum number of counts required for a cell to pass filtering.
- `min_genes`: Minimum number of genes expressed required for a cell to pass filtering.
- `max_counts`: Maximum number of counts required for a cell to pass filtering.
- `max_genes`: Maximum number of genes expressed required for a cell to pass filtering

# Keyword arguments
- `verbose`: whether to print info and status messages
- `plot_before`: whether to plot the histogram of cells before filtering
- `log_transform_plot`: whether to log-transform the counts of cells for plotting 
- `plot_after`: whether to plot the histogram of cells after filtering
- `save_plot_before`: whether to save the histogram of cells before filtering
- `save_plot_after`: whether to save the histogram of cells after filtering
- `plot_filename_before`: filename to save the histogram of cells before filtering
- `plot_filename_after`: filename to save the histogram of cells after filtering

# Returns
- the filtered `AnnData` object
"""
function filter_cells!(adata::AnnData; 
    min_counts::Union{Int, Nothing}=nothing, 
    min_genes::Union{Int, Nothing}=nothing, 
    max_counts::Union{Int, Nothing}=nothing, 
    max_genes::Union{Int, Nothing}=nothing,
    verbose::Bool = true, 
    plot_before::Bool=false,
    log_transform_plot::Bool=false,
    plot_after::Bool=false,
    save_plot_before::Bool=false, 
    save_plot_after::Bool=false,
    plot_filename_before::String="cell_histogram_before_filtering.pdf",
    plot_filename_after::String="cell_histogram_after_filtering.pdf"
    )

    options = [min_genes, min_counts, max_genes, max_counts]

    # Filter the data matrix and annotate it

    cell_subset, number_per_cell = filter_cells(adata, 
        min_counts=min_counts, min_genes=min_genes, 
        max_counts=max_counts, max_genes=max_genes, 
        verbose=verbose, 
        make_plot = plot_before,
        log_transform_plot = log_transform_plot,
        save_plot = save_plot_before,
        plot_filename = plot_filename_before
    )

    if isnothing(min_genes) && isnothing(max_genes)
        adata.obs[!,:n_counts] = number_per_cell
    else
        adata.obs[!,:n_genes] = number_per_cell
    end

    # transform BitVector to something we can subset on
    if isa(cell_subset, BitVector)
        cell_subset = collect(cell_subset)
    end

    # filter adata in place
    adata = subset_adata!(adata, cell_subset, :cells)

    # if plot_after: plot the histogram of cells after filtering
    if plot_after
        # make a histogram of the number of genes/counts per cell
        cutoff = options[.!isnothing.(options)][1]

        if !isnothing(min_genes) || !isnothing(max_genes)
            counts_number = :number
            log_transform_plot && verbose && @info "log-transforming not supported for number of genes expressed per cell, setting to false"
            log_transform_plot = false
        else
            counts_number = :counts
            if log_transform_plot 
                cutoff = log(cutoff)
            end    
        end
        
        p = plot_histogram(adata.X, :cell, counts_number; 
            cutoff = cutoff,
            log_transform = log_transform_plot,
            save_plot = save_plot_after, 
            filename = plot_filename_after
        )
        display(p)
    end

    return adata
end

filter_cells(adata::AnnData; kwargs...) = filter_cells(adata.X; kwargs...)

"""
    function filter_cells(countmatrix::AbstractMatrix; 
        min_counts::Union{Int, Nothing}=nothing, 
        min_genes::Union{Int, Nothing}=nothing, 
        max_counts::Union{Int, Nothing}=nothing, 
        max_genes::Union{Int, Nothing}=nothing,
        verbose::Bool=true,
        make_plot::Bool=false,
        log_transform_plot::Bool=false,
        save_plot::Bool=false,
        plot_filename::String="gene_filtering_histogram.pdf"
        )    
    
Filter cell outliers based on counts and numbers of genes expressed.
For instance, only keep cells with at least `min_counts` counts or
`min_genes` genes expressed. This is to filter measurement outliers,
i.e. “unreliable” observations.
Only provide one of the optional parameters `min_counts`, `min_genes`,
`max_counts`, `max_genes` per call.

# Arguments
- `countmatrix`: countmatrix of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes.
- `min_counts`: Minimum number of counts required for a cell to pass filtering.
- `min_genes`: Minimum number of genes expressed required for a cell to pass filtering.
- `max_counts`: Maximum number of counts required for a cell to pass filtering.
- `max_genes`: Maximum number of genes expressed required for a cell to pass filtering.

# Keyword arguments
- `verbose`: whether to print info and status messages
- `make_plot`: whether to plot the histogram of cells before filtering
- `log_transform_plot`: whether to log-transform the counts of cells for plotting
- `save_plot`: whether to save the histogram of cells before filtering
- `plot_filename`: filename to save the histogram of cells before filtering

# Returns
- `cells_subset`: BitVector index mask that does filtering; `true` means that the
    cell is kept, `false` means the cell is removed.
- `number_per_cell`: Depending on what was thresholded (`counts` or `genes`),
    the array stores `n_counts` or `n_genes` per cell.
"""
function filter_cells(countmatrix::AbstractMatrix; 
    min_counts::Union{Int, Nothing}=nothing, 
    min_genes::Union{Int, Nothing}=nothing, 
    max_counts::Union{Int, Nothing}=nothing, 
    max_genes::Union{Int, Nothing}=nothing,
    verbose::Bool=true,
    make_plot::Bool=false,
    log_transform_plot::Bool=false,
    save_plot::Bool=false,
    plot_filename::String="cell_filtering_histogram.pdf"
    )    
    # Check that only one filtering option is provided
    options = [min_genes, min_counts, max_genes, max_counts]
    n_given_options = sum(!isnothing(option) for option in options)
    if n_given_options != 1
        throw(ArgumentError("Only provide one of the optional parameters `min_counts`, " *
                            "`min_genes`, `max_counts`, `max_genes` per call."))
    end

    min_number = isnothing(min_genes) ? min_counts : min_genes 
    max_number = isnothing(max_genes) ? max_counts : max_genes 

    # if filtering based on min number of genes expressed, count how many genes are expressed
    if isnothing(min_genes) && isnothing(max_genes)
        number_per_cell = vec(sum(countmatrix, dims=2))
    else
        number_per_cell = vec(sum(countmatrix .> 0, dims=2))
    end

    if !isnothing(min_number)
        cell_subset = number_per_cell .>= min_number
    end

    if !isnothing(max_number)
        cell_subset = number_per_cell .<= max_number
    end
    
    if verbose # optionally: display info message 
        s = sum(.!cell_subset)
        if s > 0
            msg = "filtered out $s cells that have "
            if !isnothing(min_genes) || !isnothing(min_counts)
                msg *= "less than "
                msg *= if min_counts === nothing
                            "$min_genes genes expressed"
                        else
                            "$min_counts counts"
                        end
            end
            if !isnothing(max_genes) || !isnothing(max_counts)
                msg *= "more than "
                msg *= if max_counts === nothing
                            "$max_genes genes expressed"
                        else
                            "$max_counts counts"
                        end
            end
        else
            msg = "no cells passed the filtering threshold"
        end
        @info msg
    end

    if make_plot
        # make a histogram of the number of genes/counts per cell
        cutoff = options[.!isnothing.(options)][1]

        if !isnothing(min_genes) || !isnothing(max_genes)
            counts_number = :number
            log_transform_plot && verbose && @info "log-transforming not supported for number of genes expressed per cell, setting to false"
            log_transform_plot = false
        else
            counts_number = :counts
            if log_transform_plot 
                cutoff = log(cutoff)
            end    
        end
        
        p = plot_histogram(countmatrix, :cell, counts_number; 
            cutoff = cutoff,
            log_transform = log_transform_plot,
            save_plot = save_plot, 
            filename = plot_filename
        )
        display(p)
    end

    return cell_subset, number_per_cell
end

#-----------------------------------------------------
# gene filtering 
#-----------------------------------------------------
"""
    filter_genes!(adata::AnnData; 
        min_counts::Union{Int, Nothing}=nothing, 
        min_cells::Union{Int, Nothing}=nothing, 
        max_counts::Union{Int, Nothing}=nothing, 
        max_cells::Union{Int, Nothing}=nothing,
        verbose::Bool = true,
        plot_before::Bool=false,
        log_transform_plot::Bool=false,
        plot_after::Bool=false,
        save_plot_before::Bool=false, 
        save_plot_after::Bool=false,
        plot_filename_before::String="gene_histogram_before_filtering.pdf",
        plot_filename_after::String="gene_histogram_after_filtering.pdf"
    )
    
Filter genes based on number of cells or counts.
Keep genes that have at least `min_counts` counts or are expressed in at
least `min_cells` cells or have at most `max_counts` counts or are expressed
in at most `max_cells` cells.
Only provide one of the optional parameters `min_counts`, `min_cells`,
`max_counts`, `max_cells` per call.

# Arguments
- `adata`: `AnnData` object of shape `n_obs` × `n_vars`. Rows correspond
to cells and columns to genes.
- `min_counts`: Minimum number of counts required for a gene to pass filtering.
- `min_cells`: Minimum number of cells expressed required for a gene to pass filtering.
- `max_counts`: Maximum number of counts required for a gene to pass filtering.
- `max_cells`: Maximum number of cells expressed required for a gene to pass filtering.

# Keyword arguments
- `verbose`: whether to print info and status messages
- `plot_before`: whether to plot the histogram of genes before filtering
- `log_transform_plot`: whether to log-transform the counts of genes for plotting 
- `plot_after`: whether to plot the histogram of genes after filtering
- `save_plot_before`: whether to save the histogram of genes before filtering
- `save_plot_after`: whether to save the histogram of genes after filtering
- `plot_filename_before`: filename to save the histogram of genes before filtering
- `plot_filename_after`: filename to save the histogram of genes after filtering
    
Returns
- the filtered `AnnData` object
"""
function filter_genes!(adata::AnnData; 
    min_counts::Union{Int, Nothing}=nothing, 
    min_cells::Union{Int, Nothing}=nothing, 
    max_counts::Union{Int, Nothing}=nothing, 
    max_cells::Union{Int, Nothing}=nothing,
    verbose::Bool = true,
    plot_before::Bool=false,
    log_transform_plot::Bool=false,
    plot_after::Bool=false,
    save_plot_before::Bool=false, 
    save_plot_after::Bool=false,
    plot_filename_before::String="gene_histogram_before_filtering.pdf",
    plot_filename_after::String="gene_histogram_after_filtering.pdf"
)

    options = [min_cells, min_counts, max_cells, max_counts]
    # Filter the data matrix and annotate it

    gene_subset, number_per_gene = filter_genes(adata, 
        min_counts=min_counts, min_cells=min_cells, 
        max_counts=max_counts, max_cells=max_cells, 
        verbose=verbose,
        make_plot=plot_before,
        log_transform_plot=log_transform_plot,
        save_plot=save_plot_before,
        plot_filename=plot_filename_before
    )

    if isnothing(min_cells) && isnothing(max_cells)
        adata.var[!,:n_counts] = number_per_gene
    else
        adata.var[!,:n_cells] = number_per_gene
    end

    # transform BitVector to something we can subset on
    if isa(gene_subset, BitVector)
        gene_subset = collect(gene_subset)
    end

    # filter adata in place 
    subset_adata!(adata, gene_subset, :genes)

    # if plot_after: plot the histogram of cells after filtering
    if plot_after
        # make a histogram of the number of genes/counts per cell
        cutoff = options[.!isnothing.(options)][1]

        if !isnothing(min_cells) || !isnothing(max_cells)
            counts_number = :number
            log_transform_plot && verbose && @info "log-transforming not supported for number of cells expressed per gene, setting to false"
            log_transform_plot = false
        else
            counts_number = :counts
            if log_transform_plot 
                cutoff = log(cutoff)
            end    
        end
        
        p = plot_histogram(adata.X, :gene, counts_number; 
            cutoff = cutoff,
            log_transform = log_transform_plot,
            save_plot = save_plot_after, 
            filename = plot_filename_after
        )
        display(p)
    end

    return adata
end

filter_genes(adata::AnnData; kwargs...) = filter_genes(adata.X; kwargs...)

"""
    filter_genes(adata::AnnData; 
        min_counts::Union{Int, Nothing}=nothing, 
        min_cells::Union{Int, Nothing}=nothing, 
        max_counts::Union{Int, Nothing}=nothing, 
        max_cells::Union{Int, Nothing}=nothing,
        verbose::Bool=true,
        make_plot::Bool=false,
        log_transform_plot::Bool=false,
        save_plot::Bool=false,
        plot_filename::String="gene_filtering_histogram.pdf"
        )    
    
Filter genes based on number of cells or counts.
Keep genes that have at least `min_counts` counts or are expressed in at
least `min_cells` cells or have at most `max_counts` counts or are expressed
in at most `max_cells` cells.
Only provide one of the optional parameters `min_counts`, `min_cells`,
`max_counts`, `max_cells` per call.
This is the out-of-place version; for details on the input arguments, see the in-place version `filter_genes!`

# Arguments
- `adata`: `AnnData` object of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes.
- `min_counts`: Minimum number of counts required for a gene to pass filtering.
- `min_cells`: Minimum number of cells in which it is expressed required for a gene to pass filtering.
- `max_counts`: Maximum number of counts required for a gene to pass filtering.
- `max_cells`: Maximum number of cells in which it is expressed required for a gene to pass filtering.

# Keyword arguments
- `verbose`: whether to print info and status messages
- `make_plot`: whether to plot the histogram of cells before filtering
- `log_transform_plot`: whether to log-transform the counts of cells for plotting
- `save_plot`: whether to save the histogram of cells before filtering
- `plot_filename`: filename to save the histogram of cells before filtering

# Returns
- `gene_subset`: BitVector index mask that does filtering; `true` means that the
    gene is kept, `false` means the cell is removed.
- `number_per_gene`: Depending on what was thresholded (`counts` or `cells`),
    the array stores `n_counts` or `n_cells` per gene.
"""
function filter_genes(
    countmatrix::AbstractMatrix;
    min_counts::Union{Int, Nothing}=nothing, 
    min_cells::Union{Int, Nothing}=nothing, 
    max_counts::Union{Int, Nothing}=nothing, 
    max_cells::Union{Int, Nothing}=nothing,
    verbose::Bool=true,
    make_plot::Bool=false,
    log_transform_plot::Bool=false,
    save_plot::Bool=false,
    plot_filename::String="gene_filtering_histogram.pdf"
    )    
    # Check that only one filtering option is provided
    options = [min_cells, min_counts, max_cells, max_counts]
    n_given_options = sum(!isnothing(option) for option in options)
    if n_given_options != 1
        throw(ArgumentError("Only provide one of the optional parameters `min_counts`, " *
                            "`min_cells`, `max_counts`, `max_cells` per call."))
    end
        
    # Process the data matrix
    min_number = isnothing(min_counts) ? min_cells : min_counts
    max_number = isnothing(max_counts) ? max_cells : max_counts
    
    # Count number of cells or counts per gene
    if min_cells === nothing && max_cells === nothing
        number_per_gene = vec(sum(countmatrix, dims=1))
    else
        number_per_gene = vec(sum(countmatrix .> 0, dims=1))
    end
    
    # Apply filtering
    if !isnothing(min_number)
        gene_subset = number_per_gene .>= min_number
    end
    if !isnothing(max_number)
        gene_subset = number_per_gene .<= max_number
    end
   
    if verbose # optionally: display info message 
        s = sum(.!gene_subset)
        if s > 0
            msg = "filtered out $s genes that are detected "
            if min_cells !== nothing || min_counts !== nothing
                msg *= (min_cells === nothing ? "with less than $min_counts counts" : "in less than $min_cells cells")
            end
            if max_cells !== nothing || max_counts !== nothing
                msg *= (max_cells === nothing ? "with more than $max_counts counts" : "in more than $max_cells cells")
            end
        else
            msg = "no genes passed the filtering threshold"
        end
        @info msg
    end

    if make_plot
        # make a histogram of the number of cells/counts per gene
        cutoff = options[.!isnothing.(options)][1]
        if !isnothing(min_cells) || !isnothing(max_cells)
            counts_number = :number
            log_transform_plot && verbose && @info "log-transforming not supported for number of genes expressed per cell, setting to false"
            log_transform_plot = false
        else
            counts_number = :counts
            if log_transform_plot 
                cutoff = log(cutoff)
            end    
        end

        p = plot_histogram(countmatrix, :gene, :counts; 
            cutoff = cutoff,
            log_transform = log_transform_plot,
            save_plot = save_plot, 
            filename = plot_filename
        )
        display(p)
    end

    return gene_subset, number_per_gene

end