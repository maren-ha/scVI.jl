using scVI
using Test

@testset "scVI.jl" begin
    # Write your tests here.
    # PBMC 
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
        max_epochs=2, 
        lr = 1e-4,
        weight_decay=Float32(1e-6),
    )
    train_model!(m, adata, training_args)
    register_latent_representation!(adata, m)
end
