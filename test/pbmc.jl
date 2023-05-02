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
