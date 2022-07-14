#-------------------------------------------------------------------------------------
# Tasic data 
# for complete preprocessing: see Moritz' notebook and files
#-------------------------------------------------------------------------------------

function load_tasic(path::String = joinpath(@__DIR__, "../data/"))

    countmat = readdlm(string(path, "Tasic_countmat.txt"))
    celltypes = readdlm(string(path, "Tasic_celltypes.txt"))
    genenames = readdlm(string(path, "Tasic_genenames.txt"))
    receptorandmarkers = readdlm(string(path, "Tasic_receptorandmarkers.txt"))

    # cell type annotation: neural vs. no neural 
    celltypes = [celltypes[i,1]*" "*celltypes[i,2] for i in 1:size(celltypes,1)]
    NonNeural = ["Astro Aqp4","OPC Pdgfra","Oligo 96*Rik","Oligo Opalin","Micro Ctss","Endo Xdh","SMC Myl9"];
    neuralcells = [!(i in NonNeural) for i in celltypes];
    celltypes = [split(i)[1] for i in celltypes];

    # annotate as GABAergic vs Glutamatergic
    gabaglutamap = Dict(
        "Vip"=>"GABA",
        "L4"=>"Glutamate",
        "L2/3"=>"Glutamate",
        "L2"=>"Glutamate",
        "Pvalb"=>"GABA",
        "Ndnf"=>"GABA",
        "L5a"=>"Glutamate",
        "L5"=>"Glutamate",
        "Sst"=>"GABA",
        "L6b"=>"Glutamate",
        "Sncg"=>"GABA",
        "Igtp"=>"GABA",
        "Smad3"=>"GABA",
        "L5b"=>"Glutamate",
        "L6a"=>"Glutamate",
    )
    # encode the overall cell-type class (GABAergic vs. Glutamatergic)
    gabagluta = fill("non_neural", length(celltypes))
    gabagluta[neuralcells] = [gabaglutamap[i] for i in celltypes[neuralcells]]

    obs = Dict("cell_type" => celltypes,
                "neuralcells" => neuralcells,
                "GABAvsGluta" => gabagluta
    )

    # build final gene set
    receptorandmarker_inds = [i in receptorandmarkers for i in genenames]
    vars = Dict("gene_names" => genenames,
                "receptorandmarkers" => receptorandmarkers,
                "receptorandmarker_inds" => vec(receptorandmarker_inds)
    )

    # build layers for AnnData struct
    normalized_counts = normalizecountdata(countmat)
    layers = Dict("counts" => Float32.(countmat'),
                "normalized_counts" => Float32.(normalized_counts')
    )

    countmatrix = Float32.(countmat')
    ncells = size(countmatrix,1)
    ngenes = size(countmatrix,2)
    @assert ncells == length(obs["cell_type"])
    @assert ngenes == length(vars["gene_names"])

    adata = AnnData(
        countmatrix = countmatrix,
        ncells = ncells,
        ngenes = ngenes,
        layers=layers,
        obs=obs,
        vars=vars,
        celltypes = String.(obs["cell_type"])
    )
    return adata
end

function subset_tasic!(adata::AnnData)
    # subset to receptors and markers and neural cells only. 
    receptorandmarker_inds = adata.vars["receptorandmarker_inds"]
    neuralcells = adata.obs["neuralcells"]
    @assert size(adata.countmatrix) == (length(neuralcells), length(receptorandmarker_inds))
    adata.countmatrix = adata.countmatrix[neuralcells,receptorandmarker_inds]
    adata.ncells, adata.ngenes = size(adata.countmatrix,1), size(adata.countmatrix,2)
    adata.celltypes = String.(adata.celltypes[neuralcells])
    adata.obs = Dict(
        "cell_type" => adata.obs["cell_type"][neuralcells],
        "GABAvsGluta" => adata.obs["GABAvsGluta"][neuralcells]
    )
    adata.vars = Dict(
        "gene_names" => adata.vars["gene_names"][receptorandmarker_inds]
    )
    return adata
end