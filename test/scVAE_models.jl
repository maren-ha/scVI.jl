# ZINB distribution
@info "testing scVAE model training with ZINB distribution..."
using scVI
@info "loading data..."
path = joinpath(@__DIR__, "../data")
adata = load_tasic(path)
orig_adata = deepcopy(adata)

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
    weight_decay=Float32(1e-6)
)
train_model!(m, adata, training_args)
@test m.is_trained == true

@info "now trying with encoding lib size..."
adata.obs.batch = rand(1:2, size(adata.X,1))
library_log_means, library_log_vars = init_library_size(adata, batch_key=:batch)
m = scVAE(size(adata.X,2);
        library_log_means=library_log_means,
        library_log_vars=library_log_vars,
        n_latent=2, 
        use_observed_lib_size = false
)
print(summary(m))
training_args = TrainingArgs(
    max_epochs=2, 
    register_losses=true
)
train_model!(m, adata, training_args, batch_key=:batch)
@test !isnothing(m.l_encoder)
@test length(m.loss_registry["kl_l"]) == 2
@test sum(m.loss_registry["kl_l"]) > 0

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

@info "testing evaluations..."

@info "testing adding a latent representation..."
register_latent_representation!(adata, m)

@info "testing visualizations..."
@test haskey(adata.obsm, "scVI_latent")
plot_umap_on_latent(m, adata);
@test haskey(adata.obsm, "scVI_latent_umap")
plot_pca_on_latent(m, adata);

@info "testing adding dimension reduction + plotting with automatic registration of the latent representation..."

adata = deepcopy(orig_adata)
m = scVAE(size(adata.X,2);
    gene_likelihood = :nb,
    dispersion = :gene_cell,
    n_latent = 10,
)
train_model!(m, adata, TrainingArgs(max_epochs=2))
register_umap_on_latent!(adata, m)

adata = deepcopy(orig_adata)
train_model!(m, adata, TrainingArgs(max_epochs=2))
plot_umap_on_latent(m, adata);

adata = deepcopy(orig_adata)
train_model!(m, adata, TrainingArgs(max_epochs=2))
plot_pca_on_latent(m, adata);

adata = deepcopy(orig_adata)
train_model!(m, adata, TrainingArgs(max_epochs=2))
scVI.plot_latent_representation(m, adata);

adata = deepcopy(orig_adata)
m = scVAE(size(adata.X,2);
    gene_likelihood = :nb,
    dispersion = :gene_cell,
    n_latent = 2,
)
train_model!(m, adata, TrainingArgs(max_epochs=2))
scVI.plot_latent_representation(m, adata);

@info "testing sampling from prior and posterior..."

@info "testing sampling from posterior for negative binomial..."
m = scVAE(size(adata.X,2);
    gene_likelihood = :nb,
    n_latent = 2
)
samp = sample_from_posterior(m, adata)
@test size(samp) == (size(adata.X',1), size(adata.X',2))
@test isa(samp, Matrix{Int64})

@info "now with zero-inflated negative binomial..."
m = scVAE(size(adata.X,2);
    gene_likelihood = :zinb,
    n_latent = 2
)
samp = sample_from_posterior(m, adata)
@test size(samp) == (size(adata.X',1), size(adata.X',2))
@test isa(samp, Matrix{Int64})

@info "now with poisson..."
m = scVAE(size(adata.X,2);
    gene_likelihood = :poisson,
    n_latent = 2
)
samp = sample_from_posterior(m, adata)
@test size(samp) == (size(adata.X',1), size(adata.X',2))
@test isa(samp, Matrix{Int64})

@info "now with something that's not implemented to catch the error..."
m = scVAE(size(adata.X,2);
    n_latent = 2
)
m.gene_likelihood = :not_implemented
try
    samp = sample_from_posterior(m, adata)
catch e
    @test e == ArgumentError("Not implemented")
end

@info "testing prior sampling..."
m = scVAE(size(adata.X,2);
    gene_likelihood = :nb,
    n_latent = 2
)
priorsample = sample_from_prior(m, adata, 10)
train_model!(m, adata, TrainingArgs(max_epochs=2))
priorsample = sample_from_prior(m, adata, 10)

