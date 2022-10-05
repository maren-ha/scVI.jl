using scVI
using Revise
using DelimitedFiles: include
using Base: Float32, func_for_method_checked, String
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
using BSON
using ProgressMeter
using SpecialFunctions # for loggamma
using Statistics
using Parameters 

# For logging with tensorboard
using TensorBoardLogger: TBLogger, tb_overwrite, set_step!, set_step_increment!
using Logging: with_logger
using CUDA

# evaluation: UMAP, PCA and plots  
using LinearAlgebra
using UMAP 
using VegaLite
using Plots
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
# TODOfix the multimodal to have it train multi or unimodality, parameterized.  
# TODOturn this file into a test unit 
# TODOmerge the master in 
# Write a tutorial & python script to get the paired modalities
#################################################################
path_to_multi = "./data_sampled/multi_subsample_5000_cells_rep_0_st.h5ad"

# get the data 
@info "data loaded, initialising objects... "
multi_adata = init_benchmarking_from_h5ad(path_to_multi)

mod1 = multi_adata.countmatrix[:,1:4000]
mod2 = multi_adata.countmatrix[:,4001:4134]

@info "Benchmarking data is loaded ... "
@info "GEX modality contains $(size(multi_adata.countmatrix,1)) cells and $(size(mod1,2)) genes"
@info "Protein modality contains $(size(multi_adata.countmatrix,1)) cells and $(size(mod2,2)) proteins"

isdir("./src/runs/") || mkdir("./src/runs/")
############ Folders for experiments documentation ###############
timestamp = Dates.format(now(),"dd_mm_yyyy_HHMM")
remarks = "multi_scvi_cleaned_data"
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

#library_log_means, library_log_vars = init_library_size(adata1, 1)
#n_inputs = [size(adata1.countmatrix,2),size(adata2.countmatrix,2)]

library_log_means = 0 # we dont estimate the library size 
n_inputs = [size(mod1,2), size(mod2,2)]
training_args = TrainingArgs(
    max_epochs=25, 
    lr = 1e-3,
    weight_decay=Float32(1e-6),
    n_epochs_kl_warmup=12,
    progress = false,
    savepath=logging_path,
    verbose=true
)   

@info "Your model will run with the following parameters  
epochs: $(training_args.max_epochs) 
learning rate: $(training_args.lr)
warming up KL: $(training_args.n_epochs_kl_warmup)
tensorboard logging folder: $(training_args.savepath) 
batch_size: $(training_args.batchsize)"

scvMulti =  scMultiVAE_(n_inputs;
            n_latent=10,
            dispersion=[:gene, :gene_cell],  # :gene GEX, :gene_cell: protein 
            gene_likelihood=:zinb, 
            protein_likelihood=:nb,
            latent_distribution = :normal,
            library_log_means=library_log_means,)
model = scvMulti
#----------------------------------------------------------------
# Create a Tensorboard logger∏
#-----------------------------------------------------------------
logger = TBLogger(training_args.savepath, tb_overwrite)

x = [multi_adata]
model, adata , losses = start_training!(model, x, training_args,logger)
moes_loss, loss_rnas, loss_proteins = losses.moes_train, losses.mod1_train, losses.mod2_train;
# plot & save the losses
plot_losses(training_args.max_epochs+1,moes_loss,loss_rnas,loss_proteins, figures_path)
multi_adata = register_multilatent_representation!(multi_adata, model)
multi_adata = register_umap_on_multilatent!(multi_adata, model)
umap_plot_mod1, umap_plot_mod2, umap_plot_integrated = plot_umap_on_mixlatent(model,multi_adata; save_plot=true,figure_path=figures_path)
# save the latentspace as a csv file
# add the latent space to obms of the original data for scbi evaluation
# retreive the latentspace either from adata2.scVI_mixlatent or adata1.scVI_mixlatent, and append the obs
#integrated_latent = hcat((adata2.obs["_index"]),adata2.scVI_mixlatent')