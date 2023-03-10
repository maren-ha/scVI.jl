module DataProcessing 

# for data handling
using CSV
using DataFrames
using HDF5
using DelimitedFiles
using JLD2
using StatsBase
using LinearAlgebra

include("AnnData.jl")
include("FileIO.jl")
include("HighlyVariableGenes.jl")
include("LibrarySizeNormalization.jl")
include("Transformations.jl")
include("Cortex.jl")
include("PBMC.jl")
include("Tasic.jl")

export 
    AnnData,
    subset_adata, subset_adata!,
    read_h5ad, write_h5ad,
    init_library_size,
    highly_variable_genes, highly_variable_genes!, subset_to_hvg!,
    estimate_size_factors, normalize_size_factors, normalize_size_factors!, 
    normalize_total!, normalize_total, rescale!, log_transform!, sqrt_transform!,
    load_cortex_from_h5ad, load_cortex_from_url, load_cortex, 
    load_pbmc, 
    load_tasic, subset_tasic!
# 
end
