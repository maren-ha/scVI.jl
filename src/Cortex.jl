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
    init_cortex_from_h5ad(filename::String="cortex_anndata.h5ad")

Opens a connection to the HDF5 file saved at `filename` that stores the corresponding Python `AnnData` object created and used with `scvi-tools`. \n
Reads cortex data from an `AnnData` object created and used with the Python scVI and saved as HDF5 file. \n
Information is extracted from the file with the `load_cortex_from_h5ad` function. \n
Uses this information to fill the fields of a Julia `AnnData` object and returns it.
"""
function init_cortex_from_h5ad(filename::String="cortex_anndata.h5ad")
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

function init_cortex_from_url(save_path::String=""; verbose::Bool=false)

    url = "https://storage.googleapis.com/linnarsson-lab-www-blobs/blobs/cortex/expression_mRNA_17-Aug-2014.txt"
    path_to_file = joinpath(save_path, "expression.bin")
    if !isfile(path_to_file)
        verbose && @info "downloading data..."
        download(url, path_to_file)
    end
    verbose && @info "reading file..."
    csvfile = DelimitedFiles.readdlm(path_to_file, '\t')
    verbose && @info "extracting info..."
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

    verbose && @info "populating AnnData object..."
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

"""
    load_cortex(path::String=""; verbose::Bool=false)

Loads `cortex` dataset from [Zeisel et al. 2015](https://www.science.org/doi/10.1126/science.aaa1934) and creates a corresponding `AnnData` object. 

Looks for a file `cortex_anndata.h5ad` that can be downloaded from [this GoogleDrive `data` folder](https://drive.google.com/drive/folders/1JYNypxWnQhigEJ37jOiEwv7fzGW71jC8?usp=sharing). 
The functions first looks in the folder passed as `path` (default: assumes files are in a subfolder named `data` of the current directory, i.e., that the complete
GoogleDrive `data` folder has been downloaded in the current directory), and alternatively downloads the data if is cannot find the file in the given `path` (see below).

The file is the `h5` export of the Python `AnnData` object provided as [built-in `cortex` dataset from `scvi-tools`](https://github.com/scverse/scvi-tools/blob/master/scvi/data/_built_in_data/_cortex.py), 
data is from [Zeisel et al. 2015](https://www.science.org/doi/10.1126/science.aaa1934).

If the file is present, the data is loaded from the Python `AnnData` object and stored in an analogous Julia `AnnData` object. 
This is handled by the functions `init_cortex_from_h5ad` and `load_cortex_from_h5ad`. 

Alternatively, if the `h5ad` file is not found in the folder, the data is downloaded directly 
[from the original authors](https://storage.googleapis.com/linnarsson-lab-www-blobs/blobs/cortex/expression_mRNA_17-Aug-2014.txt) and 
processed analogous to the [`scvi-tools` processing](https://github.com/scverse/scvi-tools/blob/master/scvi/data/_built_in_data/_cortex.py), 
and subsequently stored to a Julia `AnnData` object. This is handled by the function `init_cortex_from_url`. 

Returns the Julia `AnnData` object.

**Example** 
---------------------------
    julia> load_cortex()
        AnnData object with a countmatrix with 3005 cells and 1200 genes
        layers dict with the following keys: ["counts"]
        summary statistics dict with the following keys: ["n_labels", "n_vars", "n_batch", "n_continuous_covs", "n_cells", "n_proteins"]
        unique celltypes: ["interneurons", "pyramidal SS", "pyramidal CA1", "oligodendrocytes", "microglia", "endothelial-mural", "astrocytes_ependymal"]
        training status: not trained
"""
function load_cortex(path::String="data/"; verbose::Bool=false)
    filename = joinpath(path, "cortex_anndata.h5ad")
    if isfile(filename)
        adata = init_cortex_from_h5ad(filename)
    else
        adata = init_cortex_from_url(path, verbose=verbose)
    end
    return adata 
end