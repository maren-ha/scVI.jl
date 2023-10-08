# Data processing

## `AnnData` object and I/O

The `AnnData` struct is imported from [`Muon.jl`](https://github.com/scverse/Muon.jl). The package provides read and write functions for `.h5ad` and `.h5mu` files, the typical H5-based format for storing Python `anndata` objects. The `AnnData` object stores datasets together with metadata, such as information on the variables (genes in scRNA-seq data) and observations (cells), as well as different kinds of annotations and transformations of the original count matrix, such as PCA or UMAP embeddings, or graphs of observations or variables. 

For details on the Julia implementation in `Muon.jl`, see the [documentation](https://scverse.github.io/Muon.jl/dev/).

For more details on the original Python implementation of the `anndata` object, see the [documentation](https://anndata.readthedocs.io/en/latest/) and [preprint](https://doi.org/10.1101/2021.12.16.473007).

```@docs
get_celltypes
```

```@docs
subset_adata(adata::AnnData, subset_inds::Tuple, dims::Symbol=:both)
```

```@docs
subset_adata(adata::AnnData, subset_inds::Union{Int, Vector{Int}, UnitRange, Vector{Bool}}, dims::Symbol)
```

```@docs
subset_adata!(adata::AnnData, subset_inds, dims::Symbol)
```

## Library size and normalization

```@docs
init_library_size
```

```@docs
estimate_size_factors
```

```@docs
normalize_size_factors
```

```@docs
normalize_size_factors!
```

```@docs
normalize_total!
```

```@docs
normalize_total
```

## Filtering

```@docs
filter_cells
```

```@docs
filter_cells!
```

```@docs
filter_genes!
```

```@docs
filter_genes
```

## Simple transformations 

```@docs
log_transform!
```

```@docs
logp1_transform!
```

```@docs
sqrt_transform!
```

```@docs
rescale!
```

## Dimension reduction 

```@docs
pca!
```

```@docs
umap!
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

## Plotting 

```@docs
plot_pca
```

```@docs
plot_umap
```

```@docs
plot_histogram
```

```@docs
highest_expressed_genes
```

```@docs
plot_highly_variable_genes
```

## Loading built-in datasets 

There are currently three datasets for which the package supplies built-in convenience functions for loading, processing and creating corresponding `AnnData` objects. They can be downloaded from this [Google Drive `data` folder](https://drive.google.com/drive/folders/1JYNypxWnQhigEJ37jOiEwv7fzGW71jC8?usp=sharing). The folder contains all three datasets, namely 

 *  the `cortex` data, corresponding to the [`cortex` dataset from the `scvi-tools`](https://github.com/scverse/scvi-tools/blob/master/scvi/data/_built_in_data/_cortex.py) from [Zeisel et al. 2015](https://www.science.org/doi/10.1126/science.aaa1934). The original data can be found [here](https://storage.googleapis.com/linnarsson-lab-www-blobs/blobs/cortex/expression_mRNA_17-Aug-2014.txt) and has been processed analogous to the [`scvi-tools` processing](https://github.com/scverse/scvi-tools/blob/master/scvi/data/_built_in_data/_cortex.py)
 * the `tasic` data from [Tasic et al. (2016)](https://www.nature.com/articles/nn.4216), available at [Gene expression Omnibus (GEO)](https://www.ncbi.nlm.nih.gov/geo/) under accession number GSE71585. Preprocessing and additional annotation according to the original manuscript; annotations are available and loaded together with the countmatrix. 
 * the `pbmc` data (PBMC8k) from [Zheng et al. 2017](https://www.nature.com/articles/ncomms14049), preprocessed according to the [Bioconductor workflow](https://bioconductor.org/books/3.15/OSCA.workflows/unfiltered-human-pbmcs-10x-genomics.html).

I recommend downloading the complete GoogleDrive folder and having it as a subfolder named `data` in the current working directory. Then, in any Julia script in the parent directory, the functions `load_cortex()`, `load_pbmc()` and `load_tasic()` can be called without arguments, because the default `path` where these functions look for the respective dataset is exactly that subfolder named `data`.  

### Cortex data 

```@docs
load_cortex_from_h5ad
```

```@docs 
load_cortex
```

### PBMC data 

```@docs 
load_pbmc
```

### Tasic data 

```@docs 
load_tasic
```

```@docs 
subset_tasic!
```
