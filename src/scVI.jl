module scVI

# for data handling
using CSV
using DataFrames
using HDF5

# core package functionality 
using Distributions
using Flux
using Random 
using ProgressMeter

# evaluation: UMAP, PCA and plots  
using LinearAlgebra
using UMAP 
using VegaLite

include("DataProcessing.jl")
include("Utils.jl")
include("EncoderDecoder.jl")
include("scVAEmodel.jl")
include("NegativeBinomial.jl")
include("ModelFunctions.jl")
include("Training.jl")
include("Evaluate.jl")

export 
    init_data_from_h5ad, init_library_size,
    load_cortex, load_pbmc,
    scVAE, 
    TrainingArgs, 
    train_model!, 
    get_latent_representation, 
    register_latent_representation!, register_umap_on_latent!
    plot_umap_on_latent, 
    plot_pca_on_latent
# 
end
