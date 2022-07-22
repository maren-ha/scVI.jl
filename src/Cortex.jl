#-------------------------------------------------------------------------------------
# cortex data 
#-------------------------------------------------------------------------------------

"""
    load_cortex_from_h5ad(anndata::HDF5.File)

Reads cortex data from an `AnnData` object created and used with the Python `scvi-tools` and saved as HDF5 file. 

Returns 
    the count matrix \n
    the contents of `adata.layers` \n
    the contents of `adata.obs` \n
    the contents of `adata["uns"]["_scvi"]["summary_stats"]` \n
    the contents of `adata["uns"]["_scvi"]["data_registry"]` \n
    the cell type information in `adata["obs"]["__categories"]["cell_type"]`
"""
function load_cortex_from_h5ad(anndata::HDF5.File)
    countmatrix = read(anndata, "layers")["counts"]' # shape: cell x gene 
    summary_stats = read(anndata, "uns")["_scvi"]["summary_stats"]
    layers = read(anndata, "layers")
    obs = read(anndata, "obs")
    data_registry = read(anndata, "uns")["_scvi"]["data_registry"]
    celltype_numbers = read(anndata, "obs")["cell_type"] .+1 # for Julia-Python index conversion
    celltype_categories = read(anndata, "obs")["__categories"]["cell_type"]
    celltypes = celltype_categories[celltype_numbers]
    return Matrix(countmatrix), layers, obs, summary_stats, data_registry, celltypes
end

# assumes Python adata object 
"""
    init_cortex_from_h5ad(filename::String=joinpath(@__DIR__, "../data/cortex_anndata.h5ad"))

Opens a connection to the HDF5 file saved at `filename` that stores the corresponding Python `AnnData` object created and used with `scvi-tools`. \n
Reads cortex data from an `AnnData` object created and used with the Python scVI and saved as HDF5 file. \n
Information is extracted from the file with the `load_cortex_from_h5ad` function. \n
Uses this information to fill the fields of a Julia `AnnData` object; returns the Julia `AnnData`` object.
"""
function init_cortex_from_h5ad(filename::String=joinpath(@__DIR__, "../data/cortex_anndata.h5ad"))
    anndata = open_h5_data(filename)
    countmatrix, layers, obs, summary_stats, data_registry, celltypes = load_cortex_from_h5ad(anndata)
    ncells, ngenes = size(countmatrix)
    adata = AnnData(
        countmatrix=countmatrix,
        ncells=ncells,
        ngenes=ngenes,
        layers=layers,
        obs=obs,
        summary_stats=summary_stats,
        registry=data_registry,
        celltypes=celltypes
    )
    return adata
end

function init_cortex_from_url(save_path::String=joinpath(@__DIR__, "../data/"))

    url = "https://storage.googleapis.com/linnarsson-lab-www-blobs/blobs/cortex/expression_mRNA_17-Aug-2014.txt"
    path_to_file = joinpath(save_path, "expression.bin")
    if !isfile(path_to_file)
        download(url, path_to_file)
    end
    csvfile = DelimitedFiles.readdlm(path_to_file, '\t')
    precise_clusters = csvfile[2,3:end]
    clusters = csvfile[9,3:end]
    gene_names = String.(csvfile[12:end,1])

    countmatrix = Float32.(csvfile[12:end,3:end]')

    labels = fill(0, length(clusters))
    for i in 1:length(unique(clusters))
        labels[findall(x -> x == unique(clusters)[i], clusters)] .= i
    end

    cellinfos = Dict(
        "cell_type" => clusters,
        "labels" => labels,
        "precise_labels" => precise_clusters,
        "tissue" => String.(csvfile[1,3:end]),
        "group" => Int.(csvfile[2,3:end]),
        "totalmRNA" => Int.(csvfile[3,3:end]),
        "well" => Int.(csvfile[4,3:end]),
        "sex" => Int.(csvfile[5,3:end]),
        "age" => Int.(csvfile[6,3:end]),
        "diameter" => Float32.(csvfile[7,3:end]),
        "cell_id" => String.(csvfile[8,3:end])
    )

    geneinfos = Dict(
        "gene_names" => gene_names
    )

    @assert size(countmatrix,1) == length(clusters)
    @assert size(countmatrix,2) == length(gene_names)

    adata = AnnData(
        countmatrix = countmatrix,
        ncells = size(countmatrix,1),
        ngenes = size(countmatrix,2),
        obs = cellinfos, 
        vars = geneinfos, 
        celltypes = cellinfos["cell_type"]
    )
    return adata
end

function load_cortex(path::String=joinpath(@__DIR__, "../data/"))
    if isfile(string(path, "cortex_anndata.h5ad"))
        adata = init_cortex_from_h5ad(string(path, "cortex_anndata.h5ad"))
    else
        adata = init_cortex_from_url(path)
    end
    return adata 
end