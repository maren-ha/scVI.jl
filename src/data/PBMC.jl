#-------------------------------------------------------------------------------------
# pbmc data from csv 
#-------------------------------------------------------------------------------------
"""
    load_pbmc(path::String = "data/")

Loads `pbmc` dataset from [Zheng et al. 2017](https://www.nature.com/articles/ncomms14049) and creates a corresponding `AnnData` object. 
Specifically, the PBMC8k version is used, preprocessed according to the [Bioconductor workflow](https://bioconductor.org/books/3.15/OSCA.workflows/unfiltered-human-pbmcs-10x-genomics.html).

Loads the following files that can be downloaded from [this GoogleDrive `data` folder](https://drive.google.com/drive/folders/1JYNypxWnQhigEJ37jOiEwv7fzGW71jC8?usp=sharing): 
- `PBMC_counts.csv`: countmatrix  
- `PBMC_annotation.csv`: cell type annotation

Files are loaded from the folder passed as `path` (default: assumes files are in a subfolder named `data` of the current directory, i.e., that the complete
GoogleDrive `data` folder has been downloaded in the current directory.)

From these input files, a Julia `AnnData` object is created. The countmatrix contains information on 
cell barcodes and gene names. The gene name and celltype information is stored in the `vars` and `obs` 
dictionaries of the `AnnData` object, respectively. 

Returns the Julia `AnnData` object.

**Example** 
---------------------------
    julia> load_pbmc()
        AnnData object with a countmatrix with 7480 cells and 200 genes
        unique celltypes: ["B-cells", "CD4+ T-cells", "Monocytes", "CD8+ T-cells", "NK cells", "NA", "HSC", "Erythrocytes"]
        training status: not trained
#
"""
function load_pbmc(path::String = "data/")
    filename_counts = joinpath(path, "PBMC_counts.csv")
    filename_annotation = joinpath(path, "PBMC_annotation.csv")
    if isfile(filename_counts) && isfile(filename_annotation)
        counts = CSV.read(filename_counts, DataFrame)
        celltypes = vec(string.(CSV.read(filename_annotation, DataFrame)[:,2]))
        genenames = string.(counts[:,1])
        barcodes = names(counts)[2:end]
        counts = Matrix(counts[:,2:end])
        @assert length(celltypes) == length(barcodes) == size(counts,2)
        counts = Float32.(counts')

        adata = AnnData(countmatrix = counts, 
                    celltypes = celltypes,
                    obs = DataFrame(cell_type = celltypes),
                    var = DataFrame(gene_names = genenames), 
                    layers = Dict("counts" => counts)
        )
    else
        filename_jld2 = joinpath(path, "pbmc.jld2")
        filename_in_scvi = string(dirname(pathof(scVI)),"/../data/pbmc.jld2")
        # saved via save(filename_jld2, Dict("adata_pbmc" => adata)); save(filename_in_scvi, Dict("adata_pbmc" => adata))
        if isfile(filename_jld2)
            adata = jldopen(filename_jld2)["adata_pbmc"]
        elseif isfile(filename_in_scvi)
            adata = jldopen(filename_in_scvi)["adata_pbmc"]
        end
    end
    return adata
end