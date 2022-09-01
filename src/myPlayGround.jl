using Pkg
Pkg.activate(".")
using DelimitedFiles: include
using Base: Float32
# just for the sake of testing .... 

using Dates
using Dates: print
using Random 

using CSV
using DataFrames
using HDF5
using DelimitedFiles

# core package functionality 
using Distributions
using Flux
using Flux.Data: DataLoader
using BSON: @load
using BSON: @save
using ProgressMeter
using SpecialFunctions # for loggamma
using Statistics
using Parameters 


# For logging with tensorboard
using TensorBoardLogger: TBLogger, tb_overwrite
using Logging: with_logger

# evaluation: UMAP, PCA and plots  
using LinearAlgebra
using UMAP 
using VegaLite

include("DataProcessing.jl")
include("Utils.jl")
include("EncoderDecoder.jl")
include("scVAEmultimodel.jl")
include("scVAEmodel.jl")
include("scLDVAE.jl")
include("CountDistributions.jl")
include("ModelFunctions.jl")
include("Training.jl")
include("Evaluate.jl")
include("scmmVAE.jl")

# TODO 
#################################################################
# fix the ann data to have gene and protein also the show method.
# fix the multimodal to have a vector of inputs (gex, protein) instread of redundant data. 
# fix the multimodal to have it train multi or unimodality, parameterized. 


# paths to the sampled data
path_to_gex = "./data_sampled/adata_cite_gex_subsample_5000_cells_rep_0_dense.h5ad"
path_to_protein = "./data_sampled/adata_cite_protein_subsample_5000_cells_rep_0_dense.h5ad"


# get the data 
@info "data loaded, initialising objects... "
adata1 = init_benchmarking_from_h5ad(path_to_gex)
adata2 = init_benchmarking_from_h5ad(path_to_protein) 


library_log_means, library_log_vars = init_library_size(adata1, 1)
isdir("./src/runs/") || mkdir("./src/runs/")
############ Folders for experiments documentation ###############
timestamp = Dates.format(now(),"dd_mm_yyyy_HHMM")
remarks = "multi_scvi"
isdir("./src/runs/experiment_$(remarks)_$(timestamp)") || mkdir("./src/runs/experiment_$(remarks)_$(timestamp)")
experiment_path = "./src/runs/experiment_$(remarks)_$(timestamp)"
isdir("$(experiment_path)/log/") || mkdir("$(experiment_path)/log/")
@info "The experiment folder is :" experiment_path
logging_path = "$(experiment_path)/log/"
isdir("$(experiment_path)/figures/") || mkdir("$(experiment_path)/figures/")
figures_path = "$(experiment_path)/figures/"
####################################################################################

training_args = TrainingArgs(
    max_epochs=50, 
    lr = 1e-4,
    weight_decay=Float32(1e-6),
    n_epochs_kl_warmup=12,
    progress = true,
    verbose_freq = 1, 
    log_path=logging_path,
    verbose=true
)

scvMulti =  scMultiVAE(size(adata1.countmatrix,2);
            library_log_means=library_log_means,
            n_input_2 = size(adata2.countmatrix,2),
            n_latent=10, 
            gene_likelihood=:zinb, 
            protein_likelihood=:nb,
            latent_distribution = :normal)
m = scvMulti
#----------------------------------------------------------------
# Create a Tensorboard logger
#-----------------------------------------------------------------
logger = TBLogger(training_args.log_path, tb_overwrite)
#----------------------------------------------------------------
# Train the multimodal scvMulti
#-----------------------------------------------------------------
train_multimodel!(m, adata1, adata2, training_args,logger)

# save the trained model 
#@save "$(experiment_path)/model.bson" m 
register_multilatent_representation!(adata1,adata2, m)

# Plotting 
register_umap_on_multilatent!(adata1,adata2, m)

umap_plot_mod1, umap_plot_mod2, umap_plot_mix = plot_umap_on_mixlatent(m, adata1, adata2; save_plot=true,figure_path=figures_path)