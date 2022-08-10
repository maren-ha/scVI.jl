using DelimitedFiles: include
using Base: Float32, func_for_method_checked
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

# paths to the datsa 
path_to_protein = "./data_sampled/adata_cite_protein_subsample_5000_cells_rep_0_dense.h5ad"
path_to_gex = "./data_sampled/adata_cite_gex_subsample_5000_cells_rep_0_dense.h5ad"

# get the data 
@info " Initialising data objects... "
adata1 = init_benchmarking_from_h5ad(path_to_gex)
adata2 = init_benchmarking_from_h5ad(path_to_protein)

@info "Benchmarking data is loaded ... "
@info "GEX modality contains $(size(adata1.countmatrix,1)) cells and $(size(adata1.countmatrix,2)) genes"
@info "Protein modality contains $(size(adata2.countmatrix,1)) cells and $(size(adata2.countmatrix,2)) proteins"

isdir("./src/runs/") || mkdir("./src/runs/")
############ Folders for experiments documentation ###############
timestamp = Dates.format(now(),"dd_mm_yyyy_HHMM")
remarks = "test_multi_scvi"
isdir("./src/runs/experiment_$(remarks)_$(timestamp)") || mkdir("./src/runs/experiment_$(remarks)_$(timestamp)")
experiment_path = "./src/runs/experiment_$(remarks)_$(timestamp)"
isdir("$(experiment_path)/log/") || mkdir("$(experiment_path)/log/")
@info "The experiment folder is :" experiment_path
logging_path = "$(experiment_path)/log/"
@info "Tensorboard logging folder is :" logging_path
isdir("$(experiment_path)/figures/") || mkdir("$(experiment_path)/figures/")
figures_path = "$(experiment_path)/figures/"
@info "Plots will be saved in:" figures_path
####################################################################################

library_log_means, library_log_vars = init_library_size(adata1, 1)
n_inputs = [size(adata1.countmatrix,2),size(adata2.countmatrix,2)]

training_args = TrainingArgs(
    max_epochs=75, 
    lr = 1e-3,
    weight_decay=Float32(1e-6),
    n_epochs_kl_warmup=12,
    progress = false,
    verbose_freq = 1, 
    log_path=logging_path,
    verbose=true
)

@info "Your model will run with the following parameters  
epochs: $(training_args.max_epochs) 
learning rate: $(training_args.lr)
warming up KL: $(training_args.n_epochs_kl_warmup)
tensorboard logging folder: $(training_args.log_path) 
batch_size: $(training_args.batchsize)"

scvMulti =  scMultiVAE_([size(adata1.countmatrix,2),size(adata2.countmatrix,2)];
            n_latent=10,
            dispersion=[:gene, :gene_cell],  # :gene GEX, :gene_cell: protein 
            gene_likelihood=:zinb, 
            protein_likelihood=:nb,
            latent_distribution = :normal,
            library_log_means=library_log_means,)
m = scvMulti
logger = TBLogger(training_args.log_path, tb_overwrite)
# pack the data in an array 
x = [adata1, adata2]
m, adata = start_training(m, x, training_args,logger)
adata1,adata2 = register_multilatent_representation!(adata1,adata2, m)
adata1,adata2 = register_umap_on_multilatent!(adata1,adata2, m)
umap_plot_mod1, umap_plot_mod2, umap_plot_mix = plot_umap_on_mixlatent(m, adata1, adata2; save_plot=true,figure_path=figures_path)