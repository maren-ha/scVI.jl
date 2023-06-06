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
adata.layers["counts"] = adata.X

# logp1-transformation
logp1_transform!(adata, verbose=true)
@test haskey(adata.layers, "logp1_transformed")
logp1_transform!(adata, layer="counts", verbose=true)
@test haskey(adata.layers, "logp1_transformed")
log_transform!(adata)
@test adata.layers["log_transformed"] == adata.layers["logp1_transformed"]

# normalization
normalize_total!(adata)
@test haskey(adata.layers, "normalized")

# log-transformation 
log_transform!(adata)
@test haskey(adata.layers, "log_transformed")

# sqrt transformation 
sqrt_transform!(adata, verbose=true)
@test haskey(adata.layers, "sqrt_transformed")
sqrt_transform!(adata, layer = "counts", verbose=true)
@test haskey(adata.layers, "sqrt_transformed")
sqrt_transform!(adata, layer = "something_else", verbose=true)
@test haskey(adata.layers, "sqrt_transformed")

# rescaling 
rescale!(adata, layer = "counts", verbose=true)
@test haskey(adata.layers, "rescaled")
rescale!(adata, layer = "new_layer", verbose=true)
@test haskey(adata.layers, "rescaled")
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

@info "checking corner cases for PCA and UMAP..."
dummy_adata = AnnData(X=rand((1:10), 100, 10))
@test !haskey(dummy_adata.obsm, "PCA")
@test !haskey(dummy_adata.layers, "normalized")
@test !haskey(dummy_adata.layers, "log_transformed")
pca!(dummy_adata)
@test haskey(dummy_adata.obsm, "PCA")
@test haskey(dummy_adata.layers, "normalized")
@test haskey(dummy_adata.layers, "log_transformed")
# now checking only normalization but no log transformation
dummy_adata = AnnData(X=rand((1:10), 100, 10))
normalize_total!(dummy_adata)
@test !haskey(dummy_adata.obsm, "PCA")
@test !haskey(dummy_adata.layers, "log_transformed")
@test haskey(dummy_adata.layers, "normalized")
pca!(dummy_adata)
@test haskey(dummy_adata.obsm, "PCA")
@test haskey(dummy_adata.layers, "log_transformed")
saved_pcs = deepcopy(dummy_adata.obsm["PCA"])
# now checking behaviour with more PCs than genes 
pca!(dummy_adata, n_pcs = 100)
@test dummy_adata.obsm["PCA"] ≈ saved_pcs
# now checking behaviour with layer that is not there 
pca!(dummy_adata, layer = "my_custom_layer")
@test dummy_adata.obsm["PCA"] ≈ saved_pcs

# now do a similar thing for UMAP 
dummy_adata = AnnData(X=rand((1:10), 100, 10))
@test !haskey(dummy_adata.obsm, "PCA")
@test !haskey(dummy_adata.obsm, "UMAP")
@test !haskey(dummy_adata.layers, "normalized")
@test !haskey(dummy_adata.layers, "log_transformed")

# check UMAP on PCA init, where everything still has to be calculated
umap!(dummy_adata, use_pca_init=true)
@test haskey(dummy_adata.obsm, "PCA")
@test haskey(dummy_adata.layers, "normalized")
@test haskey(dummy_adata.layers, "log_transformed")
@test haskey(dummy_adata.obsm, "umap")
@test haskey(dummy_adata.obsp, "fuzzy_neighbor_graph")
@test haskey(dummy_adata.obsm, "knns")
@test haskey(dummy_adata.obsm, "knn_dists")

# now with wrong layer specification
dummy_adata = AnnData(X=rand((1:10), 100, 10))
umap!(dummy_adata, layer = "my_custom_layer")
@test !haskey(dummy_adata.layers, "my_custom_layer")
@test haskey(dummy_adata.layers, "normalized")
@test haskey(dummy_adata.layers, "log_transformed")
@test haskey(dummy_adata.obsm, "umap")


