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

# paths to the sampled data
#path_to_gex = "./data_sampled/adata_cite_gex_subsample_5000_cells_rep_0_dense.h5ad"
#path_to_protein = "./data_sampled/adata_cite_protein_subsample_5000_cells_rep_0_dense.h5ad"
path_to_multi = "./data_sampled/multi_subsample_5000_cells_rep_0_st.h5ad"


# get the data 
@info "data loaded, initialising objects... "
#adata1 = init_benchmarking_from_h5ad(path_to_gex)
#adata2 = init_benchmarking_from_h5ad(path_to_protein) 
multi_adata = init_benchmarking_from_h5ad(path_to_multi)

mod1 = multi_adata.countmatrix[:,1:4000]
mod2 = multi_adata.countmatrix[:,4001:4134]


#library_log_means, library_log_vars = init_library_size(adata1, 1)
library_log_means = 0 # we dont estimate the library size 
n_inputs = [size(mod1,2), size(mod2,2)]

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

scvMulti =  scMultiVAE_(n_inputs;
            n_latent=10,
            dispersion=[:gene, :gene_cell],  # :gene GEX, :gene_cell: protein 
            gene_likelihood=:zinb, 
            protein_likelihood=:nb,
            latent_distribution = :normal,
            library_log_means=library_log_means,)
model = scvMulti
#----------------------------------------------------------------
# Create a Tensorboard logger
#-----------------------------------------------------------------
logger = TBLogger(training_args.log_path, tb_overwrite)
#----------------------------------------------------------------
# Train the multimodal scvMulti
#-----------------------------------------------------------------
x = [multi_adata]

model, adata , losses = start_training!(model, x, training_args,logger)
moes_loss, loss_rnas, loss_proteins = losses.moes_train, losses.mod1_train, losses.mod2_train;

# save the trained model 
#@save "$(experiment_path)/model.bson" m 
register_multilatent_representation!(multi_adata, model; save_latent=true, experiment_path=experiment_path)

# Plotting 
register_umap_on_multilatent!(multi_adata, model)

umap_plot_mod1, umap_plot_mod2, umap_plot_mix = plot_umap_on_mixlatent(m, adata1, adata2; save_plot=true,figure_path=figures_path)