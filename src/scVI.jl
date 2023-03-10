module scVI

# for data handling
include("data/DataProcessing.jl")
using .DataProcessing

export
    AnnData,
    subset_adata, subset_adata!,
    read_h5ad, write_h5ad,
    init_library_size,
    highly_variable_genes, highly_variable_genes!, subset_to_hvg!,
    estimate_size_factors, normalize_size_factors, normalize_size_factors!, 
    load_cortex_from_h5ad, load_cortex_from_url, load_cortex, 
    load_pbmc, 
    load_tasic, subset_tasic!
#

# core package functionality 
using Distributions
using Flux
using Flux:onehotbatch
using Random 
using ProgressMeter
using SpecialFunctions # for loggamma
using StatsBase

# evaluation: UMAP, PCA and plots  
using LinearAlgebra
using UMAP 
using VegaLite

include("Utils.jl")
include("EncoderDecoder.jl")
include("scVAEmodel.jl")
include("scLDVAE.jl")
include("CountDistributions.jl")
include("ModelFunctions.jl")
include("Training.jl")
include("Evaluate.jl")

export
    scVAE, scEncoder, scDecoder, scLinearDecoder, scLDVAE,
    TrainingArgs, 
    train_model!, train_supervised_model!,
    get_latent_representation, get_loadings,
    register_latent_representation!, register_umap_on_latent!,
    plot_umap_on_latent, 
    plot_pca_on_latent,
    sample_from_prior, sample_from_posterior
#

# scvis 
include("scvis/scvis.jl")
using .scvis

export
    train_scvis_model!, 
    compute_transition_probs, compute_differentiable_transition_probs,
    tsne_repel, scvis_loss, tsne_loss
#
end