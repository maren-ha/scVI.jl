using scVI
using Test
using Dates
# For logging with tensorboard
using TensorBoardLogger: TBLogger, tb_overwrite, set_step!, set_step_increment!
using Logging: with_logger
using CUDA
using VegaLite
using UMAP 


@testset "scVI.jl" begin
    # Write your tests here.
    # PBMC 
    using scVI
    @info "loading data..."
    #adata = load_pbmc()
    # change to your path, or ask Martin for the benchmarking sampled data
    path_to_gex = "/Users/sarajamal/multigrate/data/adata_cite_GEX_subsample_5000_cells_rep_0_dense.h5ad"
    adata = init_benchmarking_from_h5ad(path_to_gex)

    library_log_means, library_log_vars = scVI.init_library_size(adata,1) 
    
    isdir("./src/runs/") || mkdir("./src/runs/")
    ############ Folders for experiments documentation ###############
    timestamp = Dates.format(now(),"dd_mm_yyyy_HHMM")
    remarks = "test_scvi_tsne"
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
            gene_likelihood=:zinb, 
            dispersion=:gene,
            train_w_tsne = true
    )
    print(summary(m))
    training_args = TrainingArgs(
        max_epochs=25, 
        lr = 1e-3,
        weight_decay=Float32(1e-6),
        savepath = logging_path,
        verbose=true,
        progress=true,
        train_w_tsne=true)
    logger = TBLogger(training_args.savepath, tb_overwrite)
    # this is an ugly adaptation so our inference can work with 1 modality & more ...
    x = [adata]
    m, adata, losses = scVI.start_training!(m, x, training_args,logger)
    if m.train_w_tsne
        adata=register_latent_representation!(adata, m,false)
        # plot - to be changed and wrapped in a function ... 
        tsne_plot = @vlplot(:point, title="TSNE latent representation", 
                            x = adata.scVI_tsne_latent[1,:], 
                            y = adata.scVI_tsne_latent[2,:], 
                            color = adata.celltypes, 
                            width = 800, height=500)
        
        adata.scVI_latent_umap = umap(adata.scVI_latent, 2; min_dist=0.3)
        umap_plot = @vlplot(:point, title="UMAP latent representation", 
        x = adata.scVI_latent_umap[1,:], 
        y = adata.scVI_latent_umap[2,:], 
        color = adata.celltypes, 
        width = 800, height=500)
        tsne(copy(adata.scVI_latent')) # expects cells x dim
    else
        register_latent_representation!([adata], m)
        plot_umap_on_latent(m, adata, filename="$(figures_path)/UMAP_on_latent_ZINB_RNA.pdf")
    end 
end
