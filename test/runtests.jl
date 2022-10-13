using scVI
using Test
using Dates
using TensorBoardLogger: TBLogger, tb_overwrite


@testset "scVI.jl" begin
    # Write your tests here.
    # PBMC 
    using scVI
    @info "loading data..."
    #adata = load_pbmc()
    path_to_gex = "./data_sampled/adata_cite_protein_subsample_5000_cells_rep_0_dense.h5ad"
    adata = init_benchmarking_from_h5ad(path_to_gex)

    library_log_means, library_log_vars = scVI.init_library_size(adata,1) 
    
    isdir("./src/runs/") || mkdir("./src/runs/")
    ############ Folders for experiments documentation ###############
    timestamp = Dates.format(now(),"dd_mm_yyyy_HHMM")
    remarks = "test_uni_scvi"
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
    m = scVAE(size(adata.countmatrix,2);
            library_log_means=library_log_means,
            n_latent=10,
            modality_likelihood=:zinb, 
            dispersion=:gene
    )
    print(summary(m))
    training_args = TrainingArgs(
        max_epochs=10, 
        lr = 1e-3,
        weight_decay=Float32(1e-6),
        log_path = logging_path,
        verbose=true,
        verbose_freq=1
    )
    logger = TBLogger(training_args.log_path, tb_overwrite)
    x = [adata]
    # calling training by model ...
    m, adata = scVI.start_training(m, x, training_args,logger)
    #train_model!(m, adata, training_args,logger)
    register_latent_representation!(adata, m)
    plot_umap_on_latent(m, adata, filename="$(figures_path)/UMAP_on_latent_ZINB_RNA.pdf")
end
