#-------------------------------------------------------------------------------------
# AnnData struct
#-------------------------------------------------------------------------------------
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

function init_library_size(adata::AnnData, n_batch::Int)
    """
    Computes and returns library size.
    Parameters
    ----------
    countmatrix
        AnnData object setup with `scvi`.
    n_batch: Number of batches.
    Returns
    -------
    Tuple of two 1 x n_batch arrays containing the means and variances of library 
    size in each batch in adata.
    If a certain batch is not present in the adata, the mean defaults to 0,
    and the variance defaults to 1. These defaults are arbitrary placeholders which
    should not be used in any downstream computation.
    """
    data = try
        Matrix(get_from_registry(adata, "X")') # countmatrix: gene x cell
    catch
        adata.countmatrix
    end
    #
    batch_indices = try 
        get_from_registry(adata, "batch_indices") .+ 1
    catch
        zeros(Int,size(data,1)) .+ 1
    end

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


#-------------------------------------------------------------------------------------
# estimate size factors and normalize (based on Seurat)
#-------------------------------------------------------------------------------------

function estimatesizefactorsformatrix(mat; locfunc=median)
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

function normalizecountdata(mat)
    sizefactors = estimatesizefactorsformatrix(mat)
    return mat ./ sizefactors'
end
