# Data processing

## `AnnData` object 

```@docs 
AnnData
```

## Library size and normalization

```@docs
init_library_size(adata::AnnData)
```

```@docs
estimatesizefactorsformatrix(mat; locfunc=median)
```

```@docs
normalizecountdata(mat::AbstractMatrix)
```

```@docs
normalizecountdata!(adata::AnnData)
```

## Highly variable genes 

```@docs 
highly_variable_genes(adata::AnnData; 
    layer::Union{String,Nothing} = nothing,
    n_top_genes::Int=2000,
    batch_key::Union{String,Nothing} = nothing,
    span::Float64=0.3
    )
```

```@docs 
highly_variable_genes!(adata::AnnData; 
    layer::Union{String,Nothing} = nothing,
    n_top_genes::Int=2000,
    batch_key::Union{String,Nothing} = nothing,
    span::Float64=0.3
    )
```

```@docs 
subset_to_hvg!(adata::AnnData;
        layer::Union{String,Nothing} = nothing,
        n_top_genes::Int=2000,
        batch_key::Union{String,Nothing} = nothing,
        span::Float64=0.3
    )
```

## Loading built-in datasets 

There are currently three built-in datasets: 

 *  the `cortex` data, corresponding to the [`cortex` dataset from the `scvi-tools`](https://github.com/scverse/scvi-tools/blob/master/scvi/data/_built_in_data/_cortex.py) from [Zeisel et al. 2015](https://www.science.org/doi/10.1126/science.aaa1934).
 * the `tasic` data from [Tasic et al. (2016)](https://www.nature.com/articles/nn.4216), available at [Gene expression Omnibus (GEO)](https://www.ncbi.nlm.nih.gov/geo/) under accession number GSE71585. Preprocessing and additional annotation according to the original manuscript; annotations are available and loaded together with the countmatrix. 
 * the `pbmc` data (PBMC8k) from [Zheng et al. 2017](https://www.nature.com/articles/ncomms14049), preprocessed according to the [Bioconductor workflow](https://bioconductor.org/books/3.15/OSCA.workflows/unfiltered-human-pbmcs-10x-genomics.html).

They are stored in the repository using [Git LFS](https://git-lfs.github.com). After downloading and installing it as descriped on the website, the files can be downloaded after cloning the package repo with `git-lfs checkout`.

### Cortex data 

```@docs 
init_cortex_from_h5ad(filename::String=joinpath(@__DIR__, "../data/cortex_anndata.h5ad"))
```

```@docs 
load_cortex(path::String=joinpath(@__DIR__, "../data/"))
```

### PBMC data 

### Tasic data 

```@docs 
load_tasic(path::String = joinpath(@__DIR__, "../data/"))
```

```@docs 
subset_tasic!(adata::AnnData)
```
