Base.@kwdef mutable struct AnnData
    countmatrix::Union{Matrix,Nothing}=nothing # shape: cells by genes 
    ncells::Union{Int,Nothing}=nothing
    ngenes::Union{Int,Nothing}=nothing
    layers::Union{Dict,Nothing}=nothing
    obs::Union{Dict,Nothing}=nothing
    summary_stats::Union{Dict,Nothing}=nothing
    registry::Union{Dict,Nothing}=nothing 
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

open_h5_data(filename::String; mode::String="r+") = h5open(filename, mode)

function load_data_from_h5ad(anndata::HDF5.File)
    countmatrix = read(anndata, "layers")["counts"]' # shape: cell x gene 
    summary_stats = read(anndata, "uns")["_scvi"]["summary_stats"]
    layers = read(anndata, "layers")
    obs = read(anndata, "obs")
    data_registry = read(anndata, "uns")["_scvi"]["data_registry"]
    celltype_numbers = read(anndata, "obs")["cell_type"] .+1 # for Julia-Python index conversion
    celltype_categories = read(anndata, "obs")["__categories"]["cell_type"]
    celltypes = celltype_categories[celltype_numbers]
    return Matrix(countmatrix), layers, obs, summary_stats, data_registry, celltypes
end

# assumes Python adata object 
function init_data_from_h5ad(filename::String=joinpath(@__DIR__, "data/cortex_anndata.h5ad"))
    anndata = open_h5_data(filename)
    countmatrix, layers, obs, summary_stats, data_registry, celltypes = load_data_from_h5ad(anndata)
    ncells, ngenes = size(countmatrix)
    adata = AnnData(
        countmatrix=countmatrix,
        ncells=ncells,
        ngenes=ngenes,
        layers=layers,
        obs=obs,
        summary_stats=summary_stats,
        registry=data_registry,
        celltypes=celltypes
    )
    return adata
end

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
        @info size(data,2)  
        idx_batch = findall(batch_indices.==i_batch)
        data_batch = data[idx_batch,:]
        sum_counts = vec(sum(data_batch, dims=2))
        masked_log_sum = log.(sum_counts[findall(sum_counts.>0)])

        library_log_means[i_batch] = mean(masked_log_sum)
        library_log_vars[i_batch] = var(masked_log_sum)
    end
    return library_log_means, library_log_vars
end # to check: scvi.model._utils._init_library_size(pydata, n_batch)

function load_pbmc(path::String = joinpath(@__DIR__, "../data/"))
    counts = CSV.read(string(path, "PBMC_counts.csv"), DataFrame)
    celltypes = vec(string.(CSV.read(string(path, "PBMC_annotation.csv"), DataFrame)[:,:x]))
    genenames = string.(counts[:,1])
    barcodes = names(counts)[2:end]
    counts = Matrix(counts[:,2:end])
    @assert length(celltypes) == length(barcodes) == size(counts,2)
    counts = Float32.(counts')

    adata = scVI.AnnData(countmatrix=counts, 
                ncells=size(counts,1), 
                ngenes=size(counts,2), 
                celltypes = celltypes
    )
    return adata
end


function load_cortex(path::String=joinpath(@__DIR__, "../data/"))
    adata = init_data_from_h5ad(string(path, "cortex_anndata.h5ad"))
    return adata 
end