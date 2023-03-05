#-------------------------------------------------------------------------------------
# AnnData struct
#-------------------------------------------------------------------------------------
"""
    mutable struct AnnData

Minimal Julia implementation of the Python `AnnData` object (see [package documentation](https://anndata.readthedocs.io/en/latest/)
and [Github repository](https://github.com/scverse/anndata)): An `AnnData` object stores a `countmatrix` together with annotations 
of observations `obs` (`obsm`, `obsp`), variables `var` (`varm`, `varp`), and unstructured annotations `uns`.

**Keyword arguments** 
---------------------
 - `countmatrix::Matrix`: countmatrix in (cell x gene) shape
 - `layers::Union{Dict,Nothing}=nothing`: dictionary of other layers (e.g., normalized counts) in the same shape as the countmatrix
 - `obs::Union{DataFrame,Nothing}=nothing`: dataframe of information about cells, e.g., celltypes
 - `obsm::Union{Dict, Nothing}=nothing`: dictionary of observation-level matrices, e.g., a UMAP embedding. The first dimension of the matrix has to correspond to the number of cells.
 - `obsp::Union{Dict, Nothing}=nothing`: dictionary of (observation x observation) matrices, e.g., representing cell graphs. 
 - `var::Union{DataFrame, Nothing}=nothing`: dataframe of information about genes/features, e.g., gene names or highly variable genes
 - `varm::Union{DataFrame, Nothing}=nothing`: dictionary of variable-level matrices. The first dimension of the matrix has to correspond to the number of genes.
 - `obsp::Union{Dict, Nothing}=nothing`: dictionary of (variable x variable) matrices, e.g., representing gene graphs. 
 - `celltypes=nothing`: vector of cell type names, shorthand for `adata.obs["cell_type"]`
 - `uns::Union{Dict, Nothing}=nothing`: dictionary of unstructured annotation. 

 **Example**
 ------------------
    julia> adata = load_tasic("scvi/data/")
        AnnData object with a countmatrix with 1679 cells and 15119 genes
            layers dict with the following keys: ["normalized_counts", "counts"]
            unique celltypes: ["Vip", "L4", "L2/3", "L2", "Pvalb", "Ndnf", "L5a", "SMC", "Astro", "L5", "Micro", "Endo", "Sst", "L6b", "Sncg", "Igtp", "Oligo", "Smad3", "OPC", "L5b", "L6a"]
"""
Base.@kwdef mutable struct AnnData
    countmatrix::Matrix # shape: cells by genes 
    layers::Union{Dict,Nothing}=nothing
    obs::Union{DataFrame,Nothing}=nothing
    obsm::Union{Dict,Nothing}=nothing
    obsp::Union{Dict,Nothing}=nothing
    var::Union{DataFrame, Nothing}=nothing
    varm::Union{Dict, Nothing}=nothing
    varp::Union{Dict, Nothing}=nothing
    uns::Union{Dict, Nothing}=nothing
    celltypes=nothing
    #is_trained::Bool=false
end

Base.size(a::AnnData) = size(a.countmatrix)
Base.size(a::AnnData, ind) = size(a.countmatrix, ind)
ncells(a::AnnData) = size(a, 1)
ngenes(a::AnnData) = size(a, 2)

function Base.show(io::IO, a::AnnData)
    println(io, "$(typeof(a)) object with a countmatrix with $(ncells(a)) cells and $(ngenes(a)) genes")
    !isnothing(a.layers) && println(io, "   layers dict with the following keys: $(keys(a.layers))")
    !isnothing(a.obs) && println(io, "   information about cells: $(first(a.obs,3))")
    !isnothing(a.var) && println(io, "   information about genes: $(first(a.var,3))")
    !isnothing(a.celltypes) && println(io, "   unique celltypes: $(unique(a.celltypes))")
    #a.is_trained ? println(io, "    training status: trained") : println(io, "   training status: not trained")
    nothing 
end

# subsetting and copying 

#adata = read_h5ad("cortex_julia_anndata.h5ad")
#adata[1:10, 1:20]
#adata[2,3]

import Base.getindex

function getindex(adata::AnnData, inds...)
    adata_sub = subset_adata(adata, (inds[1], inds[2]))
end

function subset_adata(adata::AnnData, subset_inds::Tuple, dims::Symbol=:both)
    adata_new = copy(adata)
    subset_adata!(adata_new, subset_inds, dims)
    return adata_new
end

function subset_adata(adata::AnnData, subset_inds::Union{Int, Vector{Int}, UnitRange}, dims::Symbol)
    adata_new = copy(adata)
    subset_adata!(adata_new, subset_inds, dims)
    return adata_new
end

function subset_adata!(adata::AnnData, subset_inds, dims::Symbol)
    return subset_adata!(adata, subset_inds, Val(dims))
end

function subset_adata!(adata::AnnData, subset_inds::Tuple, ::Val{:both})
    subset_adata!(adata, subset_inds[1], :cells)
    subset_adata!(adata, subset_inds[2], :genes)
    return adata
end

subset_adata!(adata::AnnData, subset_inds::Tuple, ::Val{:cells}) = subset_adata!(adata, subset_inds[1], :cells)

subset_adata!(adata::AnnData, subset_inds::Tuple, ::Val{:genes}) = subset_adata!(adata, subset_inds[2], :genes)

function subset_adata!(adata::AnnData, subset_inds::Union{Int, Vector{Int}, UnitRange}, ::Val{:cells})
    #adata.ncells = length(subset_inds)
    adata.countmatrix = adata.countmatrix[subset_inds,:]
    if !isnothing(adata.celltypes)
        adata.celltypes = adata.celltypes[subset_inds]
    end
    if !isnothing(adata.obs)
        adata.obs = adata.obs[subset_inds,:]
    end
    if !isnothing(adata.layers)
        for layer in keys(adata.layers)
            adata.layers[layer] = adata.layers[layer][subset_inds,:]
        end
    end
    if !isnothing(adata.obsm)
        for key in keys(adata.obsm)
            adata.obsm[key] = adata.obsm[key][subset_inds,:]
        end
    end
    if !isnothing(adata.obsp)
        for key in keys(adata.obsp)
            adata.obsp[key] = adata.obsp[key][subset_inds,subset_inds]
        end
    end
    return adata
end

function subset_adata!(adata::AnnData, subset_inds::Union{Int, Vector{Int}, UnitRange}, ::Val{:genes})
    adata.countmatrix = adata.countmatrix[:,subset_inds]
    if !isnothing(adata.var)
        adata.var = adata.var[subset_inds,:]
    end
    if !isnothing(adata.layers)
        for layer in keys(adata.layers)
            adata.layers[layer] = adata.layers[layer][:,subset_inds]
        end
    end
    if !isnothing(adata.varm)
        for key in keys(adata.varm)
            adata.varm[key] = adata.varm[key][:,subset_inds]
        end
    end
    if !isnothing(adata.varp)
        for key in keys(adata.varp)
            adata.varp[key] = adata.varp[key][subset_inds,subset_inds]
        end
    end
    return adata
end

import Base.copy

function copy(adata::AnnData)
    bdata = AnnData(countmatrix = copy(adata.countmatrix))
    for field in fieldnames(AnnData)
        curval = getfield(adata, field)
        if !isnothing(curval)
            setfield!(bdata, field, copy(curval))
        end
    end
    return bdata
end