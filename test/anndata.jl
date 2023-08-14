adata = AnnData(
    X = [2 0; 0 1; 10 2], 
    obs = DataFrame(cell_type = ["a", "b", "c"]),
    var = DataFrame(gene_name = ["gene1", "gene2"]),
    obsm = Dict("dim_red" => [1 2; 3 4; 5 6]),
    varm = Dict("gene_vars" => [1 2 3; 3 4 5]),
    varp = Dict("gene_corrs" => [1 2; 3 4]),
    obsp = Dict("cell_corrs" => [1 2 3; 3 4 6; 5 6 7])
)
orig_adata = deepcopy(adata)

@test scVI.ncells(adata) == 3
@test scVI.ngenes(adata) == 2

subset_adata!(adata, [1,3], :cells)
bdata = subset_adata(orig_adata, [1,3], :cells)
@test scVI.ncells(adata) == 2
@test scVI.ncells(bdata) == 2

adata = deepcopy(orig_adata)
subset_adata!(adata, [1,1], :genes)
bdata = subset_adata(orig_adata, [1,1], :genes)
@test scVI.ngenes(adata) == 2
@test scVI.ngenes(bdata) == 2

show(orig_adata)

rename!(adata.obs, :cell_type => :celltype)
@test get_celltypes(adata) == ["a", "b", "c"]
rename!(adata.obs, :celltype => :celltypes)
@test get_celltypes(adata) == ["a", "b", "c"]
rename!(adata.obs, :celltypes => :cell_types)
@test get_celltypes(adata) == ["a", "b", "c"]