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
