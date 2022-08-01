#-------------------------------------------------------------------------------------
# AnnData struct
#-------------------------------------------------------------------------------------
"""
    mutable struct AnnData

Minimal Julia implementation of the Python `AnnData` object (see [package documentation](https://anndata.readthedocs.io/en/latest/)
and [Github repository](https://github.com/scverse/anndata)).

**Keyword arguments** 
---------------------
 - `countmatrix::Union{Matrix,Nothing}=nothing`: countmatrix in cell x gene shape
 - `ncells::Union{Int,Nothing}=nothing`: number of cells; `size(countmatrix,1)`
 - `ngenes::Union{Int,Nothing}=nothing`: number of genes; `size(countmatrix,2)`
 - `layers::Union{Dict,Nothing}=nothing`: dictionary of other layers (e.g., normalized counts), corresponds to `adata.layers`
 - `obs::Union{Dict,Nothing}=nothing`: dictionary of information about cells, e.g., celltypes
 - `summary_stats::Union{Dict, Nothing}=nothing`: dictionary of summary information, corresponds to `adata.uns["_scvi"]["summary_stats"]`
 - `registry::Union{Dict, Nothing}=nothing`: dictionary corresponding to `adata.uns["_scvi"]["data_registry"]`
 - `vars::Union{Dict, Nothing}=nothing`: dictionary of information about genes/features, e.g., gene names or highly variable genes
 - `celltypes=nothing`: vector of cell type names, shorthand for `adata.obs["cell_type"]`
 - `train_inds=nothing`: vector of cell indices used for training an `scVAE` model 
 - `dataloader=nothing`: `Flux.DataLoader` object used for training an `scVAE` model
 - `scVI_latent=nothing`: latent representation of trained `scVAE` model 
 - `scVI_latent_umap=nothing`: UMAP of latent representation from trained `scVAE` model 
 - `is_trained::Bool=false`: indicating whether a model has been trained on the AnnData object 

 **Example**
 ------------------
    julia> adata = load_tasic("scvi/data/")
        AnnData object with a countmatrix with 1679 cells and 15119 genes
            layers dict with the following keys: ["normalized_counts", "counts"]
            unique celltypes: ["Vip", "L4", "L2/3", "L2", "Pvalb", "Ndnf", "L5a", "SMC", "Astro", "L5", "Micro", "Endo", "Sst", "L6b", "Sncg", "Igtp", "Oligo", "Smad3", "OPC", "L5b", "L6a"]
            training status: not trained
"""
Base.@kwdef mutable struct AnnData
    countmatrix::Union{Matrix,Nothing}=nothing # shape: cells by genes 
    ncells::Union{Int,Nothing}=nothing
    ngenes::Union{Int,Nothing}=nothing
    layers::Union{Dict,Nothing}=nothing
    obs::Union{Dict,Nothing}=nothing
    summary_stats::Union{Dict,Nothing}=nothing
    registry::Union{Dict,Nothing}=nothing
    vars::Union{Dict, Nothing}=nothing
    celltypes=nothing
    train_inds=nothing
    dataloader=nothing
    scVI_latent=nothing
    scVI_latent_umap=nothing
    is_trained::Bool=false
end

function Base.show(io::IO, a::AnnData)
    println(io, "$(typeof(a)) object with a countmatrix with $(a.ncells) cells and $(a.ngenes) genes")
    !isnothing(a.layers) && println(io, "   layers dict with the following keys: $(keys(a.layers))")
    !isnothing(a.summary_stats) && println(io, "   summary statistics dict with the following keys: $(keys(a.summary_stats))")
    !isnothing(a.celltypes) && println(io, "   unique celltypes: $(unique(a.celltypes))")
    a.is_trained ? println(io, "    training status: trained") : println(io, "   training status: not trained")
    nothing 
end

#-------------------------------------------------------------------------------------
# general functions 
#-------------------------------------------------------------------------------------

open_h5_data(filename::String; mode::String="r+") = h5open(filename, mode)

function get_from_registry(adata::AnnData, key)
    data_loc = adata.registry[key]
    attr_name, attr_key = data_loc["attr_name"], data_loc["attr_key"]
    data = getfield(adata, Symbol(attr_name))[attr_key]
    return data
end

"""
    init_library_size(adata::AnnData)

Computes and returns library size based on `AnnData` object. \n
Based on the `scvi-tools` function from [here](https://github.com/scverse/scvi-tools/blob/04389f74f3e94d7d2986f93eac85cb4543a8608f/scvi/model/_utils.py#L229) \n
Returns a tupe of arrays of length equal to the number of batches in `adata` as stored in `adata.obs["batch_indices"]`, 
containing the means and variances of the library size in each batch in `adata`. If no batch indices are given, defaults to 1 batch.     
"""
function init_library_size(adata::AnnData)
    data = adata.countmatrix
    #
    if !isnothing(adata.obs) && haskey(adata.obs, "batch")
        batch_indices = adata.obs["batch"]
    else
        batch_indices = try 
            get_from_registry(adata, "batch_indices") .+ 1 # for Python-Julia index conversion 
        catch
            ones(Int,size(data,1)) 
        end
    end

    n_batch = length(unique(batch_indices))

    library_log_means = zeros(n_batch)
    library_log_vars = ones(n_batch)

    for i_batch in unique(batch_indices)
        # @info size(data,2)  
        idx_batch = findall(batch_indices.==i_batch)
        data_batch = data[idx_batch,:]
        sum_counts = vec(sum(data_batch, dims=2))
        masked_log_sum = log.(sum_counts[findall(sum_counts.>0)])

        library_log_means[i_batch] = mean(masked_log_sum)
        library_log_vars[i_batch] = var(masked_log_sum)
    end
    return library_log_means, library_log_vars
end # to check: scvi.model._utils._init_library_size(pydata, n_batch)

#-------------------------------------------------------------------------------------
# get highly variable genes 
# from scanpy: https://github.com/scverse/scanpy/blob/master/scanpy/preprocessing/_highly_variable_genes.py
# not yet fully equivalent to Python (difference: 18 genes)
#-------------------------------------------------------------------------------------

using Loess
using StatsBase

function check_nonnegative_integers(X::AbstractArray) 
    if eltype(X) == Integer
        return true 
    elseif any(sign.(X) .< 0)
        return false 
    elseif !(all(X .% 1 .≈ 0))
        return false 
    else
        return true 
    end
end

# expects batch key in "obs" Dict
# results are comparable to scanpy.highly_variable_genes, but differ slightly. 
# when using the Python results of the Loess fit though, genes are identical. 
function _highly_variable_genes_seurat_v3(adata::AnnData; 
    layer::Union{String,Nothing} = nothing,
    n_top_genes::Int=2000,
    batch_key::Union{String,Nothing} = nothing,
    span::Float64=0.3,
    inplace::Bool=true
    )
    X = !isnothing(layer) ? adata.layers[layer] : adata.countmatrix
    !check_nonnegative_integers(X) && @warn "flavor Seurat v3 expects raw count data, but non-integers were found"

    means, vars = mean(X, dims=1), var(X, dims=1)
    batch_info = isnothing(batch_key) ? zeros(size(X,1)) : adata.obs[batch_key]
    norm_gene_vars = []
    for b in unique(batch_info)
        X_batch = X[findall(x -> x==b, batch_info),:]
        m, v = vec(mean(X_batch, dims=1)), vec(var(X_batch, dims=1))
        not_const = vec(v .> 0)
        estimat_var = zeros(size(X,2))
        y = Float64.(log10.(v[not_const]))
        x = Float64.(log10.(m[not_const]))
        loess_model = loess(x, y, span=span, degree=2);
        fitted_values = Loess.predict(loess_model,x)
        estimat_var[not_const] = fitted_values
        reg_std = sqrt.(10 .^estimat_var)

        batch_counts = copy(X_batch)
        N = size(X_batch,1)
        vmax = sqrt(N)
        clip_val = reg_std .* vmax .+ m
        clip_val_broad = vcat([clip_val' for _ in 1:size(batch_counts,1)]...)
        mask = batch_counts .> clip_val_broad
        batch_counts[findall(mask)] .= clip_val_broad[findall(mask)]
        squared_batch_counts_sum = vec(sum(batch_counts.^2, dims=1))
        batch_counts_sum = vec(sum(batch_counts,dims=1))
        norm_gene_var = (1 ./((N-1) .* reg_std.^2)) .* ((N.*m.^2) .+ squared_batch_counts_sum .- 2 .* batch_counts_sum .* m)
        push!(norm_gene_vars, norm_gene_var)
    end
    norm_gene_vars = hcat(norm_gene_vars...)'
    # argsort twice gives ranks, small rank means most variable
    ranked_norm_gene_vars = mapslices(row -> sortperm(sortperm(-row)), norm_gene_vars,dims=2)
    # this is done in SelectIntegrationFeatures() in Seurat v3
    num_batches_high_var = sum(mapslices(row -> row .< n_top_genes, ranked_norm_gene_vars, dims=2), dims=1)
    ranked_norm_gene_vars = Float32.(ranked_norm_gene_vars)
    ranked_norm_gene_vars[findall(x -> x > n_top_genes, ranked_norm_gene_vars)] .= NaN
    median_ranked = mapslices(col -> mymedian(col[findall(x -> !isnan(x), col)]), ranked_norm_gene_vars, dims=1)

    sortdf = DataFrame(row = collect(1:length(vec(median_ranked))),
                    highly_variable_rank = vec(median_ranked),
                    highly_variable_nbatches = vec(num_batches_high_var)
    )
    sorted_df = sort(sortdf, [:highly_variable_rank, order(:highly_variable_nbatches, rev = true)])
    sorted_index = sorted_df[!,:row]
    highly_variable = fill(false, length(median_ranked))
    highly_variable[sorted_index[1:n_top_genes]] .= true

    hvg_info = Dict("highly_variable" => highly_variable,
                "highly_variable_rank" => vec(median_ranked),
                "means" => means,
                "variances" => vars, 
                "variances_norm" => vec(mean(norm_gene_vars, dims=1))
    )
    if !isnothing(batch_key)
        hvg_info["highly_variable_nbatches"] = vec(num_batches_high_var)
    end

    if inplace 
        adata.vars = merge(adata.vars, hvg_info)
        return adata
    else
        return hvg_info
    end
end

mymedian(X::AbstractArray) = length(X) == 0 ? NaN : median(X)

"""
    highly_variable_genes!(adata::AnnData;
        layer::Union{String,Nothing} = nothing,
        n_top_genes::Int=2000,
        batch_key::Union{String,Nothing} = nothing,
        span::Float64=0.3
        )

Computes highly variable genes per batch according to the workflows on `scanpy` and Seurat v3 in-place. 

More specifically, it is the Julia re-implementation of the corresponding 
[`scanpy` function](https://github.com/scverse/scanpy/blob/master/scanpy/preprocessing/_highly_variable_genes.py)

For implementation details, please check the `scanpy`/Seurat documentations or the source code of the 
lower-level `_highly_variable_genes_seurat_v3` function in this package. 
Results are almost identical to the `scanpy` function. The differences have been traced back to differences in 
the local regression for the mean-variance relationship implemented in the Loess.jl package, that differs slightly 
from the corresponding Python implementation. 

**Arguments**
------------------------
- `adata`: `AnnData` object 
- `layer`: optional; which layer to use for calculating the HVGs. Function assumes this is a layer of counts. If `layer` is not provided, `adata.countmatrix` is used. 
- `n_top_genes`: optional; desired number of highly variable genes. Default: 2000. 
- `batch_key`: optional; key where to look for the batch indices in `adata.obs`. If not provided, data is treated as one batch. 
- `span`: span to use in the loess fit for the mean-variance local regression. See the Loess.jl docs for details. 

**Returns**
------------------------
This is the in-place version that adds an dictionary containing information on the highly variable genes directly 
to the `adata.vars` and returns the modified `AnnData` object. 
Specifically, a dictionary with the following keys is added: 
 - `highly_variable`: vector of `Bool`s indicating which genes are highly variable
 - `highly_variable_rank`: rank of the highly variable genes according to (corrected) variance 
 - `means`: vector with means of each gene
 - `variances`: vector with variances of each gene 
 - `variances_norm`: normalized variances of each gene 
 - `highly_variable_nbatches`: if there are batches in the dataset, logs the number of batches in which each highly variable gene was actually detected as highly variable. 
"""
function highly_variable_genes!(adata::AnnData; 
    layer::Union{String,Nothing} = nothing,
    n_top_genes::Int=2000,
    batch_key::Union{String,Nothing} = nothing,
    span::Float64=0.3
    )
    return _highly_variable_genes_seurat_v3(adata; 
                layer=layer, 
                n_top_genes=n_top_genes, 
                batch_key=batch_key,
                span=span,
                inplace=true
    )
end

"""
    highly_variable_genes(adata::AnnData;
        layer::Union{String,Nothing} = nothing,
        n_top_genes::Int=2000,
        batch_key::Union{String,Nothing} = nothing,
        span::Float64=0.3
        )

Computes highly variable genes according to the workflows on `scanpy` and Seurat v3 per batch and returns a dictionary with 
the information on the joint HVGs. For the in-place version, see `highly_variable_genes!`

More specifically, it is the Julia re-implementation of the corresponding 
[`scanpy` function](https://github.com/scverse/scanpy/blob/master/scanpy/preprocessing/_highly_variable_genes.py)
For implementation details, please check the `scanpy`/Seurat documentations or the source code of the 
lower-level `_highly_variable_genes_seurat_v3` function in this package. 
Results are almost identical to the `scanpy` function. The differences have been traced back to differences in 
the local regression for the mean-variance relationship implemented in the Loess.jl package, that differs slightly 
from the corresponding Python implementation. 

**Arguments**
------------------------
- `adata`: `AnnData` object 
- `layer`: optional; which layer to use for calculating the HVGs. Function assumes this is a layer of counts. If `layer` is not provided, `adata.countmatrix` is used. 
- `n_top_genes`: optional; desired number of highly variable genes. Default: 2000. 
- `batch_key`: optional; key where to look for the batch indices in `adata.obs`. If not provided, data is treated as one batch. 
- `span`: span to use in the loess fit for the mean-variance local regression. See the Loess.jl docs for details. 

**Returns**
------------------------
Returns a dictionary containing information on the highly variable genes, specifically containing the following keys is added: 
 - `highly_variable`: vector of `Bool`s indicating which genes are highly variable
 - `highly_variable_rank`: rank of the highly variable genes according to (corrected) variance 
 - `means`: vector with means of each gene
 - `variances`: vector with variances of each gene 
 - `variances_norm`: normalized variances of each gene 
 - `highly_variable_nbatches`: if there are batches in the dataset, logs the number of batches in which each highly variable gene was actually detected as highly variable. 
"""
function highly_variable_genes(adata::AnnData; 
    layer::Union{String,Nothing} = nothing,
    n_top_genes::Int=2000,
    batch_key::Union{String,Nothing} = nothing,
    span::Float64=0.3
    )
    return _highly_variable_genes_seurat_v3(adata; 
                layer=layer, 
                n_top_genes=n_top_genes, 
                batch_key=batch_key,
                span=span,
                inplace=false
    )
end

"""
    subset_to_hvg!(adata::AnnData;
        layer::Union{String,Nothing} = nothing,
        n_top_genes::Int=2000,
        batch_key::Union{String,Nothing} = nothing,
        span::Float64=0.3
    )

Calculates highly variable genes with `highly_variable_genes!` and subsets the `AnnData` object to the calculated HVGs. 
For description of input arguments, see `highly_variable_genes!`

Returns: `adata` object subset to the calculated HVGs, both in the countmatrix/layer data used for HVG calculation and in the `adata.vars` dictionary.
"""
function subset_to_hvg!(adata::AnnData;
    layer::Union{String,Nothing} = nothing,
    n_top_genes::Int=2000,
    batch_key::Union{String,Nothing} = nothing,
    span::Float64=0.3
    )
    if !haskey(adata.vars,"highly_variable")
        highly_variable_genes!(adata; 
            layer=layer, 
            n_top_genes=n_top_genes,
            batch_key=batch_key,
            span=span
        )
    end

    hvgs = adata.vars["highly_variable"]
    @assert size(adata.countmatrix,2) == length(hvgs)
    adata.countmatrix = adata.countmatrix[:,hvgs]
    adata.ngenes = size(adata.countmatrix,2)
    for key in keys(adata.vars)
        if length(adata.vars[key]) == length(hvgs)
            adata.vars[key] = adata.vars[key][hvgs]
        end
    end
    # some basic checks 
    @assert sum(adata.vars["highly_variable"]) == adata.ngenes
    @assert !any(isnan.(adata.vars["highly_variable_rank"]))
    return adata
end

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
    normalize_counts(mat::Abstractmatrix)

Normalizes the countdata in `mat` by dividing it by the size factors calculated with `estimate_size_factors`. 
Assumes a countmatrix `mat` in cell x gene format as input, returns the normalized matrix.
"""
function normalize_counts(mat::AbstractMatrix)
    sizefactors = estimate_size_factors(mat)
    return mat ./ sizefactors'
end

"""
    normalize_counts(adata::AnnData)

Normalizes the `adata.countmatrix` by dividing it by the size factors calculated with `estimate_size_factors`. 
Adds the normalized count matrix to `adata.layers` and returns `adata`.
"""
function normalize_counts!(adata::AnnData)
    sizefactors = estimate_size_factors(adata.countmatrix)
    mat_norm = mat ./ sizefactors'
    if !isnothing(adata.layers)
        adata.layers = Dict()
    end
    adata.layers["normalised"] = mat_norm
    return adata
end
