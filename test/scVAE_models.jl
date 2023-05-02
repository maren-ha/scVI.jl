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
# additionally: within FC layers 
layers = scVI.FCLayers(size(adata.X,2), 10; 
    n_hidden = [128, 64, 32], 
    n_layers = 2
)
@test length(layers) == 2
@test length(layers[1].layers[1].bias) == 128
@test size(layers[2].layers[1].weight) == (10, 128)
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