# PBMC
@info "testing PBMC data loading + VAE model initialization..."
using scVI

# load smaller dataset from CSV 
@info "loading smaller example dataset from CSVs in `test/data`..."
adata = load_pbmc(joinpath(@__DIR__, "data"))
@test size(adata.X) == (5,5)
@test nrow(adata.obs) == 5
@test nrow(adata.var) == 5

@info "loading larger dataset from main package `data` folder..."
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

# how to obtain test CSVs
# using CSV, DataFrames
# path = "data/"
# filename_counts = joinpath(path, "PBMC_counts.csv")
# filename_annotation = joinpath(path, "PBMC_annotation.csv")
# small_counts = CSV.read(filename_counts, DataFrame)[1:5,1:6]
# small_celltypes = CSV.read(filename_annotation, DataFrame)[1:5,:]
# CSV.write("scVI/test/data/PBMC_counts.csv", small_counts)
# CSV.write("scVI/test/data/PBMC_annotation.csv", small_celltypes)