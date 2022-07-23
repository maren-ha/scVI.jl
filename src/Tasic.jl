#-------------------------------------------------------------------------------------
# Tasic data 
# for complete preprocessing: see Moritz' notebook and files
#-------------------------------------------------------------------------------------
"""
    load_tasic(path::String = joinpath(@__DIR__, "../data/"))

Loads build-in `tasic` dataset from [Tasic et al. (2016)](https://www.nature.com/articles/nn.4216). 

Loads the following files from the folder passed as `path` (default: `data` subfolder of scVI repo.)
 - `Tasic_countmat.txt`: countmatrix  
 - `Tasic_celltypes.txt`: cell types
 - `Tasic_genenames.txt`: gene names 
 - `Tasic_receptorandmarkers.txt`: List of receptor and marker genes 
These files can be downloaded from the repo using [Git LFS](https://git-lfs.github.com) and running `git-lfs checkout`
after cloning the repository. The original data is available at [Gene expression Omnibus (GEO)](https://www.ncbi.nlm.nih.gov/geo/) under accession number GSE71585. 
Preprocessing and annotation has been prepared according to the original manuscript. 

From these input files, a Julia `AnnData` object is created. The list of receptor and marker genes is used 
to annotate cells as neural vs. non-neural, and annotate the neural cells as  GABA- or Glutamatergic. 

These annotations together with the cell type information and the gene names and receptor/marker list are stored in Dictionaries 
in the `obs` and `vars` fields of the `AnnData` obejct. 

Additionally, size factors are calculated and used for normalizing the counts. 
The normalized counts are stored in an additional `layer` named `normalized_counts`.

Returns the Julia `AnnData` object.

**Example** 
---------------------------
    julia> load_tasic()
        AnnData object with a countmatrix with 1679 cells and 15119 genes
        layers dict with the following keys: ["normalized_counts", "counts"]
        unique celltypes: ["Vip", "L4", "L2/3", "L2", "Pvalb", "Ndnf", "L5a", "SMC", "Astro", "L5", "Micro", "Endo", "Sst", "L6b", "Sncg", "Igtp", "Oligo", "Smad3", "OPC", "L5b", "L6a"]
        training status: not trained"""
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

"""
    subset_tasic!(adata::AnnData)

Subsets an input `AnnData` object initialized from the Tasic data according `load_tasic` to the neural cells 
and the receptor and marker genes provided as annotation. 

Specifically, the count matrix and the normalized count matrix are subset to these cells and genes, 
and the dictionaries with information about cells and genes in `adata.obs` and `adata.vars` are also subset accordingly. 

Returns the modified `AnnData` object. 
"""
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