"""
    init_library_size(adata::AnnData; batch_key::Symbol=:batch)

Computes and returns library size based on `AnnData` object. \n
Based on the `scvi-tools` function from [here](https://github.com/scverse/scvi-tools/blob/04389f74f3e94d7d2986f93eac85cb4543a8608f/scvi/model/_utils.py#L229) \n
Returns a tupe of arrays of length equal to the number of batches in `adata` as stored in `adata.obs[!,:batch_key]`, 
containing the means and variances of the library size in each batch in `adata`. Default batch key: `:batch`, if it is not found, defaults to 1 batch.     
"""
function init_library_size(adata::AnnData; batch_key::Symbol=:batch)
    data = adata.countmatrix
    #
    if !isnothing(adata.obs) && hasproperty(adata.obs, batch_key)
        batch_indices = adata.obs[!,batch_key]
        if 0 ∈ batch_indices
            batch_indices .+= 1 # for Julia-Python index conversion 
        end
    else
        batch_indices = ones(Int,size(data,1))
    end

    n_batch = length(unique(batch_indices))

    library_log_means = zeros(Float32, n_batch)
    library_log_vars = ones(Float32, n_batch)

    for (ind, i_batch) in enumerate(unique(batch_indices))
        # @info size(data,2)  
        idx_batch = findall(batch_indices.==i_batch)
        data_batch = data[idx_batch,:]
        sum_counts = vec(sum(data_batch, dims=2))
        masked_log_sum = log.(sum_counts[findall(sum_counts.>0)])

        library_log_means[ind] = mean(masked_log_sum)
        library_log_vars[ind] = var(masked_log_sum)
    end
    return library_log_means, library_log_vars
end # to check: scvi.model._utils._init_library_size(pydata, n_batch)

#-------------------------------------------------------------------------------------
# estimate size factors and normalize (based on Seurat)
#-------------------------------------------------------------------------------------

"""
    estimate_size_factors(mat; locfunc=median)

Estimates size factors to use for normalization, based on the corresponding Seurat functionality. 
Assumes a countmatrix `mat` in cell x gene format as input, returns a vector of size factors. 

For details, please see the Seurat documentation. 
"""
function estimate_size_factors(mat; locfunc=median)
    logcounts = log.(mat)
    loggeomeans = vec(mean(logcounts, dims=2))
    finiteloggeomeans = isfinite.(loggeomeans)
    loggeomeans = loggeomeans[finiteloggeomeans]
    logcounts = logcounts[finiteloggeomeans,:]
    nsamples = size(logcounts, 2)
    size_factors = fill(0.0, nsamples)
    for i = 1:nsamples
        size_factors[i] = exp(locfunc(logcounts[:,i] .- loggeomeans))
    end
    return size_factors
end

"""
    normalize_size_factors(mat::Abstractmatrix)

Normalizes the countdata in `mat` by dividing it by the size factors calculated with `estimate_size_factors`. 
Assumes a countmatrix `mat` in cell x gene format as input, returns the normalized matrix.
"""
function normalize_size_factors(mat::AbstractMatrix)
    sizefactors = estimate_size_factors(mat)
    return mat ./ sizefactors'
end

"""
    normalize_size_factors(adata::AnnData)

Normalizes the `adata.countmatrix` by dividing it by the size factors calculated with `estimate_size_factors`. 
Adds the normalized count matrix to `adata.layers` and returns `adata`.
"""
function normalize_size_factors!(adata::AnnData)
    sizefactors = estimate_size_factors(adata.countmatrix)
    mat_norm = mat ./ sizefactors'
    if !isnothing(adata.layers)
        adata.layers = Dict()
    end
    adata.layers["size_factor_normalized"] = mat_norm
    return adata
end

"""
    normalize_total!(adata::AnnData; 
        target_sum::Union{Nothing, Real}=nothing, 
        key_added::String="cell_counts",
        layer::Union{Nothing, String}=nothing,
        verbose::Bool=false)

Normalizes counts per cell, specifically normalize each cell by total counts over all genes,
so that every cell has the same total count after normalization.
If choosing `target_sum=1e6`, this is CPM normalization.

Basic version of the [scanpy.pp.normalize_total function](https://github.com/scverse/scanpy/blob/ed3b277b2f498e3cab04c9416aaddf97eec8c3e2/scanpy/preprocessing/_normalization.py#L45-L241)

**Arguments**
------------------------
- `adata`: `AnnData` object 
- `target_sum`: if `nothing`, after normalization, each observation (cell) has a total count equal to the median of total counts for observations (cells) before normalization.
- `key_added`: name of the field in `adata.obs` where the normalization factor is stored, set to "cell_counts" by default 
- `layer`: optional; which layer to normalize on. If `nothing`, `adata.countmatrix` is used. 

**Returns**
-------
Returns `adata` with normalized version of the original `adata.X` in `adata.layers` and the size factors in `adata.obs`. 
"""
function normalize_total!(adata::AnnData; 
            target_sum::Union{Nothing, Real}=nothing, 
            key_added::String="cell_counts",
            layer::Union{Nothing, String}=nothing,
            verbose::Bool=false
    )
    X, scaled_counts_per_cell = _normalize_total(adata; target_sum=target_sum, layer=layer, verbose=verbose)
    verbose && @info "adding size factors to adata.obs..."
    adata.obs[!,key_added] = vec(scaled_counts_per_cell)
    if isnothing(adata.layers)
        adata.layers = Dict()
    end
    verbose && @info "adding normalized counts to adata.layers..."
    adata.layers["normalized"] = X 
    return adata
end 

"""
    normalize_total(adata::AnnData; 
        target_sum::Union{Nothing, Real}=1e4, 
        key_added::String="cell_counts",
        verbose::Bool=false)

Normalizes counts per cell, specifically normalize each cell by total counts over all genes,
so that every cell has the same total count after normalization. See `normalize_total!` for details. 
Unlike the in-place version, this function returns a dictionary with the normalized counts and scaled counts per cell. 
"""
function normalize_total(adata::AnnData; 
            target_sum::Union{Nothing, Real}=1e4, 
            layer::Union{Nothing, String}=nothing,
            verbose::Bool=false
    )
    X, scaled_counts_per_cell = _normalize_total(adata; target_sum=target_sum, layer=layer, verbose=verbose)
    return Dict("X" => X, "scaled_counts_per_cell" => scaled_counts_per_cell)
end 

function _normalize_total(adata::AnnData; 
            target_sum::Union{Nothing, Real}=1e4,
            layer::Union{Nothing, String}=nothing,
            verbose::Bool=false
    )
    X = isnothing(layer) ? adata.countmatrix : adata.layers[layer]
    verbose && @info "normalizing counts per cell..."
    counts_per_cell = sum(X, dims=2)

    counts_greater_zero = counts_per_cell[counts_per_cell .> 0]
    if isnothing(target_sum)
        target_sum = median(counts_greater_zero)
    end
    #counts += counts == 0
    norm_X  = (X ./ vec(counts_per_cell)) .* target_sum
    return norm_X, counts_per_cell ./ target_sum 
end
