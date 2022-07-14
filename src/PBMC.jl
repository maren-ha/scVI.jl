#-------------------------------------------------------------------------------------
# pbmc data from csv 
#-------------------------------------------------------------------------------------

function load_pbmc(path::String = joinpath(@__DIR__, "../data/"))
    counts = CSV.read(string(path, "PBMC_counts.csv"), DataFrame)
    celltypes = vec(string.(CSV.read(string(path, "PBMC_annotation.csv"), DataFrame)[:,:x]))
    genenames = string.(counts[:,1])
    barcodes = names(counts)[2:end]
    counts = Matrix(counts[:,2:end])
    @assert length(celltypes) == length(barcodes) == size(counts,2)
    counts = Float32.(counts')

    adata = AnnData(countmatrix=counts, 
                ncells=size(counts,1), 
                ngenes=size(counts,2), 
                celltypes = celltypes,
                obs=Dict("cell_type" => celltypes),
                vars = Dict("gene_names" => genenames)
    )
    return adata
end
