#include("AnnData.jl")

#-------------------------------------------------------------------------------------
# get highly variable genes 
# from scanpy: https://github.com/scverse/scanpy/blob/master/scanpy/preprocessing/_highly_variable_genes.py
# not yet fully equivalent to Python (difference: 18 genes)
#-------------------------------------------------------------------------------------

using Loess
using StatsBase

is_nonnegative_integer(x::Integer) = x ≥ 0
is_nonnegative_integer(x) = false

function check_nonnegative_integers(X::AbstractArray) 
    if eltype(X) == Integer
        return all(is_nonnegative_integer.(X)) 
    elseif any(sign.(X) .< 0)
        return false 
    elseif !(all(X .% 1 .≈ 0))
        return false 
    else
        return true 
    end
end

# expects batch key in "obs" dataframe
# results are comparable to scanpy.highly_variable_genes, but differ slightly. 
# when using the Python results of the Loess fit though, genes are identical. 
function _highly_variable_genes_seurat_v3(adata::AnnData; 
    layer::Union{String,Nothing} = nothing,
    n_top_genes::Int=2000,
    batch_key::Union{Symbol, String, Nothing} = nothing,
    span::Float64=0.3,
    inplace::Bool=true,
    replace_hvgs::Bool=true,
    verbose::Bool=false
    )
    X = !isnothing(layer) ? adata.layers[layer] : adata.X
    !check_nonnegative_integers(X) && @warn "flavor Seurat v3 expects raw count data, but non-integers were found"
    verbose && @info "input checks passed..."
    means, vars = mean(X, dims=1), var(X, dims=1)
    batch_info = isnothing(batch_key) ? zeros(size(X,1)) : adata.obs[!,Symbol(batch_key)]
    norm_gene_vars = []
    verbose && @info "calculating variances per batch..."
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
    verbose && @info "identifying top HVGs..."
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

    hvg_info = DataFrame(highly_variable = highly_variable,
                highly_variable_rank = vec(median_ranked),
                means = vec(means),
                variances = vec(vars), 
                variances_norm = vec(mean(norm_gene_vars, dims=1))
    )
    if !isnothing(batch_key)
        hvg_info[!,:highly_variable_nbatches] = vec(num_batches_high_var)
    end

    if inplace 
        if isnothing(adata.var)
            adata.var = hvg_info
        else
            if !isempty(intersect(names(hvg_info), names(adata.var))) && replace_hvgs # if there is already HVG information present and it should be replaced
                other_col_inds = findall(x -> !(x ∈ names(hvg_info)), names(adata.var)) # find indices of all columns that are not contained in the new hvg_info df
                adata.var = hcat(adata.var[!,other_col_inds], hvg_info) # keep only the cols not recalculated in the new hvg_info df, and append the hvg_info df
            else
                adata.var = hcat(adata.var, hvg_info, makeunique=true)
            end
        end
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
        span::Float64=0.3,
        replace_hvgs::Bool=true,
        verbose::Bool=false
        )

Computes highly variable genes per batch according to the workflows on `scanpy` and Seurat v3 in-place. 
This is the in-place version that adds an dictionary containing information on the highly variable genes directly 
to the `adata.var` and returns the modified `AnnData` object. 
For details, see the not-in-place version `?highly_variable_genes`. 
"""
function highly_variable_genes!(adata::AnnData; 
    layer::Union{String,Nothing} = nothing,
    n_top_genes::Int=2000,
    batch_key::Union{Symbol,String,Nothing} = nothing,
    span::Float64=0.3, 
    replace_hvgs::Bool=true,
    verbose::Bool=false
    )
    return _highly_variable_genes_seurat_v3(adata; 
                layer=layer, 
                n_top_genes=n_top_genes, 
                batch_key=batch_key,
                span=span,
                inplace=true, 
                replace_hvgs=replace_hvgs,
                verbose=verbose
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
- `layer`: optional; which layer to use for calculating the HVGs. Function assumes this is a layer of counts. If `layer` is not provided, `adata.X` is used. 
- `n_top_genes`: optional; desired number of highly variable genes. Default: 2000. 
- `batch_key`: optional; key where to look for the batch indices in `adata.obs`. If not provided, data is treated as one batch. 
- `span`: span to use in the loess fit for the mean-variance local regression. See the Loess.jl docs for details. 
- `replace_hvgs`: whether or not to replace the hvg information if there are already hvgs calculated. If false, the new values are added with a "_1" suffix. Default:true,
- `verbose`: whether or not to print info on current status

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
    batch_key::Union{String,Symbol,Nothing} = nothing,
    span::Float64=0.3,
    replace_hvgs::Bool=true,
    verbose::Bool=false
    )
    return _highly_variable_genes_seurat_v3(adata; 
                layer=layer, 
                n_top_genes=n_top_genes, 
                batch_key=batch_key,
                span=span,
                inplace=false,
                replace_hvgs=replace_hvgs,
                verbose=verbose
    )
end

"""
    subset_to_hvg!(adata::AnnData;
        layer::Union{String,Nothing} = nothing,
        n_top_genes::Int=2000,
        batch_key::Union{String,Nothing} = nothing,
        span::Float64=0.3,
        verbose::Bool=true
    )

Calculates highly variable genes with `highly_variable_genes!` and subsets the `AnnData` object to the calculated HVGs. 
For description of input arguments, see `highly_variable_genes!`

Returns: `adata` object subset to the calculated HVGs, both in the countmatrix/layer data used for HVG calculation and in the `adata.var` dictionary.
"""
function subset_to_hvg!(adata::AnnData;
    layer::Union{String,Nothing} = nothing,
    n_top_genes::Int=2000,
    batch_key::Union{String,Symbol,Nothing} = nothing,
    span::Float64=0.3,
    verbose::Bool=true
    )
    if isnothing(adata.var) || (!isnothing(adata.var) && !hasproperty(adata.var,:highly_variable))
        verbose && @info "no HVGs found, calculating highly variabls genes using flavor seurat v3 in-place..."
        highly_variable_genes!(adata; 
            layer=layer, 
            n_top_genes=n_top_genes,
            batch_key=batch_key,
            span=span,
            replace_hvgs=true,
            verbose=verbose
        )
    end

    hvgs = adata.var[!,:highly_variable]
    @assert size(adata.X,2) == length(hvgs)
    subset_inds = collect(hvgs)
    subset_adata!(adata, subset_inds, :genes)
    #adata.X = adata.X[:,hvgs]
    #adata.var = adata.var[hvgs,:]
    #adata.ngenes = size(adata.X,2)
    #for colname in names(adata.var)
    #    if length(adata.var[!,colname]) == length(hvgs)
    #        adata.var[!,colname] = adata.var[!,colname][hvgs]
    #    end
    #end
    # some basic checks 
    @assert sum(adata.var[!,:highly_variable]) == size(adata.X,2)
    @assert !any(isnan.(adata.var[!,:highly_variable_rank]))
    return adata
end