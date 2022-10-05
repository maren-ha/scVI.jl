module scVI

# for data handling
using CSV
using DataFrames
using HDF5
using DelimitedFiles

using Dates
using Dates: print

# core package functionality 
using Distributions
using Flux
using BSON
using Random 
using ProgressMeter
using SpecialFunctions # for loggamma
using StatsBase

# evaluation: UMAP, PCA and plots  
using LinearAlgebra
using UMAP 
using VegaLite

#logging 
using TensorBoardLogger: TBLogger, tb_overwrite
using Logging: with_logger
# adata_rna.obs.rename(columns = {'':'n_genes_by_counts', :''}, inplace = True)

include("DataProcessing.jl")
include("Cortex.jl")
include("PBMC.jl")
include("Tasic.jl")
include("Utils.jl")
include("EncoderDecoder.jl")
include("scVAEmodel.jl")
include("scVAEmultimodel.jl")
include("scLDVAE.jl")
include("CountDistributions.jl")
include("ModelFunctions.jl")
include("Training.jl")
include("Evaluate.jl")



export 
    AnnData,
    init_cortex_from_h5ad, init_library_size,
    highly_variable_genes, highly_variable_genes!, 
    #subset_to_hvg!,
    #estimatesizefactorsformatrix, normalizecountdata,
    load_cortex, load_pbmc, init_benchmarking_from_h5ad,load_tasic, subset_tasic!,
    scVAE, scEncoder, scDecoder, scLinearDecoder,scMultiVAE_,scLDVAE,
    TrainingArgs, 
    train_model!, 
    get_latent_representation, get_loadings,
    register_latent_representation!, register_umap_on_latent!,
    plot_umap_on_latent, 
    plot_pca_on_latent
# 
end
