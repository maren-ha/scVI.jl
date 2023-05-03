@info "testing more training functionality..."

@info "testing `setup_batch_indices_for_library_scaling`..."

using scVI 
adata = load_cortex()
subset_to_hvg!(adata, n_top_genes=500)

library_log_means, library_log_vars = init_library_size(adata)

m = scVAE(size(adata.X,2);
    library_log_means=library_log_means,
    library_log_vars=library_log_vars, 
    n_latent=2, 
    use_observed_lib_size = false
)
batch_indices = scVI.setup_batch_indices_for_library_scaling(m, adata, :batch)
@test length(unique(batch_indices)) == length(library_log_means) == 1
@test m.use_observed_lib_size == true # was changed in the function because batch key was not found 

m = scVAE(size(adata.X,2);
    use_observed_lib_size = false, 
    library_log_means=library_log_means,
    library_log_vars=library_log_vars
)
batch_indices = scVI.setup_batch_indices_for_library_scaling(m, adata, :age)
@test length(unique(batch_indices)) == length(library_log_means) == 1
@test length(unique(batch_indices)) != length(unique(adata.obs.age))
@test m.use_observed_lib_size == true # was changed in the function because batch key was not used for library size calculation 

# now do it correctly
library_log_means, library_log_vars = init_library_size(adata, batch_key = :age)
m = scVAE(size(adata.X,2);
    use_observed_lib_size = false, 
    library_log_means=library_log_means,
    library_log_vars=library_log_vars
)
batch_indices = scVI.setup_batch_indices_for_library_scaling(m, adata, :age)
@test length(unique(batch_indices)) == length(library_log_means)
@test m.n_batch == length(library_log_means)
@test m.use_observed_lib_size == false

# now with m.use_observed_lib_size = true
m = scVAE(size(adata.X,2);
    use_observed_lib_size = true, 
    library_log_means=library_log_means,
    library_log_vars=library_log_vars
)
batch_indices = scVI.setup_batch_indices_for_library_scaling(m, adata, :age)
@test length(unique(batch_indices)) == 1
@test length(unique(batch_indices)) != length(library_log_means)

@info "testing details in train function..."

@info "testing train-test split, verbose_freq instead of progress, and loss registry..."
training_args = TrainingArgs(train_test_split=true, 
    trainsize = 0.8, 
    register_losses = true, 
    max_epochs = 2, 
    verbose = true, 
    verbose_freq = 10, 
    progress = false
)
train_model!(m, adata, training_args)
@test isa(m.loss_registry, Dict)
@test length(m.loss_registry) == 4
@test length(m.loss_registry["reconstruction"]) == training_args.max_epochs
@test haskey(m.loss_registry, "reconstruction")
@test haskey(m.loss_registry, "total_loss")
@test haskey(m.loss_registry, "kl_z")
@test haskey(m.loss_registry, "kl_l")
@test sum(m.loss_registry["reconstruction"]) > 0
@test sum(m.loss_registry["total_loss"]) > 0

@info "testing supervised training..."
labels = randn(Float32, (size(adata.X,1), m.n_latent))
train_supervised_model!(m, adata, labels, training_args)
train_supervised_model!(m, adata, labels, TrainingArgs(max_epochs=1))

