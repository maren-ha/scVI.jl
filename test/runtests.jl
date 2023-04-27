using scVI
using Test
using DataFrames

@testset "Preprocessing" begin
    
    using scVI
    @info "loading data..."
    adata = load_cortex()

    # test basic AnnData funcs, filtering + subsetting
    @info "testing basic AnnData funcs, filtering + subsetting..."
    @test size(adata.X,1) == 3005
    @test size(adata.X,2) == 19972
    
    filter_cells!(adata, max_counts=10000)
    @test size(adata.X,1) == 1205

    filter_genes!(adata, min_counts=10)
    @test size(adata.X,2) == 14385

    subset_adata!(adata, collect(1:100), :cells)
    @test size(adata.X,1) == 100
    
    subset_adata!(adata, collect(1:100), :genes)
    @test size(adata.X,2) == 100

    # reloading data 
    adata = load_cortex()

    # check transformations 
    @info "testing transformations..."
    logp1_transform!(adata)
    @test haskey(adata.layers, "logp1_transformed")
    normalize_total!(adata)
    @test haskey(adata.layers, "normalized")
    log_transform!(adata)
    @test haskey(adata.layers, "log_transformed")
    sqrt_transform!(adata)
    @test haskey(adata.layers, "sqrt_transformed")

    # check HVG selection
    @info "testing HVG selection..."
    hvgdict = highly_variable_genes(adata, n_top_genes=1200)
    @test sum(hvgdict[!,"highly_variable"]) == 1200
    highly_variable_genes!(adata, n_top_genes=1200)
    @test sum(adata.var[!,"highly_variable"]) == 1200
    subset_to_hvg!(adata, n_top_genes=1200)
    @test nrow(adata.var) == 1200

    # check PCA and UMAP
    @info "testing PCA and UMAP..."
    pca!(adata)
    @test haskey(adata.obsm, "PCA")
    umap!(adata)
    @test haskey(adata.obsm, "umap")
end

@testset "PBMC.jl" begin
    # PBMC
    using scVI
    @info "loading data..."
    adata = load_pbmc()
    @info "data loaded, initialising object... "
    library_log_means, library_log_vars = init_library_size(adata) 
    m = scVAE(size(adata.X,2);
            library_log_means=library_log_means,
            library_log_vars=library_log_vars, 
            #use_observed_lib_size=false
    )
    print(summary(m))
end

@testset "basic scVAE models" begin
    # ZINB distribution
    using scVI
    @info "loading data..."
    adata = load_pbmc()
    @info "data loaded, initialising object... "
    library_log_means, library_log_vars = init_library_size(adata) 
    m = scVAE(size(adata.X,2);
            library_log_means=library_log_means,
            n_latent=2
    )
    print(summary(m))
    training_args = TrainingArgs(
        max_epochs=1, 
        lr = 1e-4,
        weight_decay=Float32(1e-6),
    )
    train_model!(m, adata, training_args)
    register_latent_representation!(adata, m)

    # NB distribution
    m = scVAE(size(adata.X,2);
        library_log_means=library_log_means,
        gene_likelihood = :nb,
        n_latent = 5, 
    )
    print(summary(m))
    training_args = TrainingArgs(
    max_epochs=1, 
    lr = 1e-4,
    weight_decay=Float32(1e-6),
    )
    train_model!(m, adata, training_args)
end

@testset "Gaussian likelihood" begin
    adata = load_cortex()
    subset_to_hvg!(adata, n_top_genes=1200)
    normalize_total!(adata)
    log_transform!(adata)

    m = scVAE(size(adata.layers["log_transformed"], 2), 
        n_latent=2, 
        gene_likelihood=:gaussian
    )

    training_args = TrainingArgs(
        max_epochs=1, 
        weight_decay=Float32(1e-6),
        register_losses=false
    )

    train_model!(m, adata, training_args; layer = "log_transformed")
end

