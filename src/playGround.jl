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
include("scVAEmodel.jl")
include("CountDistributions.jl")
include("ModelFunctions.jl")
include("Evaluate.jl")
include("scmmVAE.jl")
include("Training_scmm.jl")


# paths to the datsa 
path_to_protein = "./data_sampled/adata_cite_protein_subsample_5000_cells_rep_0_dense.h5ad"


# get the data 
@info "data loaded, initialising objects... "
adata1 = init_benchmarking_from_h5ad(path_to_protein) 


############ Folders for experiments documentation ###############
timestamp = Dates.format(now(),"dd_mm_yyyy_HHMM")
remarks = "scmm_protein"
isdir("./src/runs/experiment_$(remarks)_$(timestamp)") || mkdir("./src/runs/experiment_$(remarks)_$(timestamp)")
experiment_path = "./src//runs/experiment_$(remarks)_$(timestamp)"
isdir("$(experiment_path)/log/") || mkdir("$(experiment_path)/log/")
@info "The experiment folder is :" experiment_path
logging_path = "$(experiment_path)/log/"
isdir("$(experiment_path)/figures/") || mkdir("$(experiment_path)/figures/")
figures_path = "$(experiment_path)/figures/"
####################################################################################

training_args = TrainingArgs(
    max_epochs=100, 
    lr = 1e-4,
    weight_decay=Float32(1e-6),
    n_epochs_kl_warmup=25,
    progress = false,
    verbose_freq = 1, 
    log_path=logging_path,
    verbose=true
)



scmm = scmmVAE(size(adata1.countmatrix,2);n_latent=10, gene_likelihood=:nb,latent_distribution = :normal)

m = scmm
#----------------------------------------------------------------
# Create a Tensorboard logger
#-----------------------------------------------------------------
logger = TBLogger(training_args.log_path, tb_overwrite)
#----------------------------------------------------------------
# Train the multimodal scvMulti
#-----------------------------------------------------------------
train_model!(m, adata1, training_args,logger)

register_latent_representation_!(adata1, m)
plot_umap_on_latent_(m, adata1, filename="$(figures_path)/UMAP_SCMM_PROTEIN.pdf")


function register_latent_representation_!(adata::AnnData, m::scmmVAE)
    adata.scVI_latent = get_latent_representation_(m, adata.countmatrix)
    @info "latent representation added"
    return adata 
end

function get_latent_representation_(m::scmmVAE, countmatrix::Matrix; cellindices=nothing, give_mean::Bool=true)
    # countmatrix assumes cells x genes 
    if !isnothing(cellindices)
        countmatrix = countmatrix[cellindices,:]
    end
    z, qz_m, qz_v = scmminference(m,countmatrix')
    if give_mean
        return qz_m
    else
        return z
    end
end

function plot_umap_on_latent_(m::scmmVAE, adata::AnnData; save_plot::Bool=true, seed::Int=111, filename::String="UMAP_on_latent.pdf")

    if isnothing(adata.scVI_latent) 
        @info "no latent representation saved in AnnData object, calculating based on scVAE model..."
        register_latent_representation_!(adata, m)
    end

    if isnothing(adata.scVI_latent_umap)
        @info "no UMAP of latent representation saved in AnnData object, calculating it now..."
        Random.seed!(seed)
        register_umap_on_latent_!(adata, m)
    end

    umap_plot = @vlplot(:point, 
                        title="UMAP of scVI latent representation", 
                        x=adata.scVI_latent_umap[1,:], 
                        y = adata.scVI_latent_umap[2,:], 
                        color = adata.celltypes, 
                        width = 800, height=500
    )
    save_plot && save(filename, umap_plot)
    return umap_plot
end

function register_umap_on_latent_!(adata::AnnData, m::scmmVAE)
    if isnothing(adata.scVI_latent)
        @info "no latent representation saved in AnnData object, calculating based on scVAE model..."
        register_latent_representation_!(adata, m)
    end
    adata.scVI_latent_umap = umap(adata.scVI_latent, 2; min_dist=0.3)
    @info "UMAP of latent representation added"
    return adata
end