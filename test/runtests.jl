using scVI
using Test
using DataFrames
using Random

@testset "Preprocessing with Cortex data" begin
    
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
    rescale!(adata)
    @test haskey(adata.layers, "rescaled")

    # check HVG selection
    @info "testing HVG selection..."
    hvgdf = highly_variable_genes(adata, n_top_genes=1200)
    @test sum(hvgdf[!,"highly_variable"]) == 1200
    hvgdf_batch = highly_variable_genes(adata, n_top_genes = 500, batch_key = "sex")
    @test hasproperty(hvgdf_batch, :highly_variable_nbatches)
    adata.var = hvgdf_batch
    @test sum(adata.var.highly_variable) == 500
    highly_variable_genes!(adata, n_top_genes=1200, replace_hvgs = true)
    @test sum(adata.var.highly_variable) == 1200
    subset_to_hvg!(adata, n_top_genes=1200)
    @test nrow(adata.var) == 1200

    # check PCA and UMAP
    @info "testing PCA and UMAP..."
    pca!(adata)
    @test haskey(adata.obsm, "PCA")
    umap!(adata)
    @test haskey(adata.obsm, "umap")
end

@testset "PBMC data" begin
    # PBMC
    @info "testing PBMC data loading + VAE model initialization..."
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
    @test m.is_trained == false

    @info "testing LDVAE model initialization..."
    m = scLDVAE(size(adata.X,2);
        library_log_means=library_log_means,
        library_log_vars=library_log_vars, 
    )
    @test hasfield(typeof(m.decoder), :factor_regressor)
end

@testset "Tasic data" begin
    @info "testing PBMC data loading + VAE model initialization..."
    using scVI 
    @info "loading data..."
    path = joinpath(@__DIR__, "../data")
    adata = load_tasic(path)
    @test size(adata.X) == (21, 1500)
    @test size(adata.obs) == (21, 3)
    @info "subset to neural cells and receptor + marker genes only..."
    subset_tasic!(adata)
    @test size(adata.X) == (15, 80)
    @test size(adata.var) == (80, 2)
end

@testset "Highly variable gene utils" begin
    @info "testing highly variable gene utils: `check_nonnegative_integers`..."
    using scVI
    @test scVI.is_nonnegative_integer(0.3) == false 
    @test scVI.is_nonnegative_integer(-6) == false
    @test scVI.is_nonnegative_integer(6) == true
    x = randn(10, 5)
    @test scVI.check_nonnegative_integers(x) == false
    x = rand(10, 5)
    @test scVI.check_nonnegative_integers(x) == false
    x = rand(-10:10, (10, 5))
    @test scVI.check_nonnegative_integers(x) == false
    x = rand(1:10, (10, 5))
    @test scVI.check_nonnegative_integers(x) == true
    x = Float32.(x)
    @test scVI.check_nonnegative_integers(x) == true
end

@testset "basic scVAE models" begin
    # ZINB distribution
    @info "testing scVAE model training with ZINB distribution..."
    using scVI
    @info "loading data..."
    path = joinpath(@__DIR__, "../data")
    adata = load_tasic(path)
    @info "data loaded, initialising object... "
    library_log_means, library_log_vars = init_library_size(adata) 
    @info "Using model with wrong number of layers to see if warning is outputted correctly and number of layers adjusted..."
    m = scVAE(size(adata.X,2);
        n_layers = 2,
        n_hidden = [128, 64, 32]
    )
    @test m.n_layers == 2
    @test m.n_hidden == 128 
    @info "Testing `use_observed_lib_size=false`..."
    try
        m = scVAE(size(adata.X,2);
            use_observed_lib_size=false
        )
    catch e
        @test isa(e, ArgumentError)
    end
    @info "Testing wrong gene likelihood specification..."
    m = scVAE(size(adata.X, 2), 
        gene_likelihood = :gamma_poisson
    )
    @test m.gene_likelihood == :nb

    @info "now trying with correctly specified model..."
    m = scVAE(size(adata.X,2);
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            n_latent=2, 
            use_observed_lib_size = false
    )
    print(summary(m))
    training_args = TrainingArgs(
        max_epochs=1, 
        lr = 1e-4,
        weight_decay=Float32(1e-6),
    )
    train_model!(m, adata, training_args)
    @test m.is_trained == true    

    # NB distribution
    @info "testing scVAE model training with NB distribution..."
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
    @test m.is_trained == true 

    @info "testing adding a latent representation..."
    register_latent_representation!(adata, m)

    @info "testing visualizations..."
    @test haskey(adata.obsm, "scVI_latent")
    plot_umap_on_latent(m, adata);
    @test haskey(adata.obsm, "scVI_latent_umap")
    plot_pca_on_latent(m, adata);
end

@testset "Gaussian + Bernoulli likelihood" begin
    @info "testing loss with Gaussian likelihood..."
    adata = load_cortex()
    subset_to_hvg!(adata, n_top_genes=200)
    #normalize_total!(adata)
    log_transform!(adata)

    m = scVAE(size(adata.layers["log_transformed"], 2), 
        n_latent=2, 
        gene_likelihood=:gaussian
    )
    lossval = scVI.loss(m, adata.X'; kl_weight=1.0f0)
    @test isa(lossval, Float32)

    @info "testing loss with Bernoulli likelihood..."
    normalize_total!(adata)
    log_transform!(adata)
    binarized = adata.layers["log_transformed"] .> 0
    adata.layers["binarized"] = Float32.(adata.layers["log_transformed"] .> 0)
    m = scVAE(size(adata.layers["binarized"], 2), 
        n_latent=2, 
        gene_likelihood=:bernoulli
    )
    @test m.gene_likelihood == :bernoulli
    lossval = scVI.loss(m, adata.layers["binarized"]'; kl_weight=1.0f0)
    @test isa(lossval, Float32)
end


