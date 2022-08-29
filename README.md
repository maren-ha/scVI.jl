# README

![](logo/scvi-julia-logo.jpg)

A Julia package for fitting VAEs to single-cell data using count distributions. 
Based on the Python implementation in the [`scvi-tools`](https://github.com/scverse/scvi-tools) package. 

## Introduction

The scVI model was first proposed in [Lopez R, Regier J, Cole MB *et al.* Deep generative modeling for single-cell transcriptomics. *Nat Methods* **15**, 1053-1058 (2018)](https://doi.org/10.1038/s41592-018-0229-2). 

More on the much more extensive Python package ecosystem `scvi-tools` can be found on the 
[website](https://scvi-tools.org) and in the corresponding paper [Gayoso A, Lopez R, Xing G. *et al.* A Python library for probabilistic analysis of single-cell omics data. *Nat Biotechnol* **40**, 163–166 (2022)](https://doi.org/10.1038/s41587-021-01206-w). 

This is the documentation for the Julia version implementing basic functionality, including the following (non-exhausive list): 

- standard and linearly decoded VAE models 
- support for negative binomial generative distribution with and without zero-inflation 
- different ways of specifying the dispersion parameter 
- store data in a (very basic) Julia version of the Python [`AnnData`](https://anndata.readthedocs.io/en/latest/) objects 
- several built-in datasets 
- training routines supporting a wide range of customisable hyperparameters

## Getting started 

The package can be downloaded from this Github repo and added with the Julia package manager via: 

```
using Pkg 
Pkg.add(url="https://github.com/maren-ha/scVI.jl")
using scVI 
```

## Built-in data

There are currently three datasets for which the package supplies built-in convenience functions for loading, processing and creating corresponding `AnnData` objects. They can be downloaded from this [Google Drive `data` folder](https://drive.google.com/drive/folders/1JYNypxWnQhigEJ37jOiEwv7fzGW71jC8?usp=sharing). The folder contains all three datasets, namely 

 *  the `cortex` data, corresponding to the [`cortex` dataset from the `scvi-tools`](https://github.com/scverse/scvi-tools/blob/master/scvi/data/_built_in_data/_cortex.py) from [Zeisel et al. 2015](https://www.science.org/doi/10.1126/science.aaa1934).
 * the `tasic` data from [Tasic et al. (2016)](https://www.nature.com/articles/nn.4216), available at [Gene expression Omnibus (GEO)](https://www.ncbi.nlm.nih.gov/geo/) under accession number GSE71585. Preprocessing and additional annotation according to the original manuscript; annotations are available and loaded together with the countmatrix. 
 * the `pbmc` data (PBMC8k) from [Zheng et al. 2017](https://www.nature.com/articles/ncomms14049), preprocessed according to the [Bioconductor workflow](https://bioconductor.org/books/3.15/OSCA.workflows/unfiltered-human-pbmcs-10x-genomics.html).

After downloading the `data` folder from Google Drive, the functions `load_cortex`, `load_tasic` and `load_pbmc` can be used to get preprocessed `adata` objects from each of these datasets, respectively (check the [docs](https://maren-ha.github.io/scVI.jl/) for more details). 

## Demo usage

The following illustrate a quick demo usage of the package. 

We load the `cortex` data (one of the built-in datasets of `scvi-tools`). If the `data` subfolder has been downloaded and the Julia process is started in the parent directory of that folder, the dataset will be loaded directly from the one supplied in `data`, if not, it will be downloaded from the original authors.

We subset the dataset to the 1200 most highly variable genes and calculate the library size. Then, we initialise a `scVAE` model and train for 50 epochs using default arguments. We visualise the results by running UMAP on the latent representation and plotting. 

```
# load cortex data
adata = load_cortex(; verbose=true)

# subset to highly variable genes 
subset_to_hvg!(adata, n_top_genes=1200, verbose=true)

# calculate library size 
library_log_means, library_log_vars = init_library_size(adata)

# initialise scVAE model 
m = scVAE(size(adata.countmatrix,2);
        library_log_means=library_log_means,
        library_log_vars=library_log_vars
)

# train model
training_args = TrainingArgs(
    max_epochs=50, # 50 for 10-dim 
    weight_decay=Float32(1e-6),
)
train_model!(m, adata, training_args)

# plot UMAP of latent representation 
plot_umap_on_latent(m, adata; save_plot=true)
```

## Docs 

[Current version of docs.](https://maren-ha.github.io/scVI.jl/)

This is the manual deployment of the local build. Automatic deployment with GithubActions/TravisCI is work in progress and will (hopefully) be available soon. 

As with every Julia package, you can access the docstrings of exported function by typing `?` into the REPL, followed by the function name. E.g., `?normalize_counts` prints the following to the REPL:

```
help?> normalize_counts
search: normalize_counts normalize_counts!

  normalize_counts(mat::Abstractmatrix)

  Normalizes the countdata in mat by dividing it by the size factors calculated with estimate_size_factors. Assumes a countmatrix mat in cell x gene format as
  input, returns the normalized matrix.
```

To reproduce the current version of the documentation or include your own extensions of the code, you can run `julia make.jl` inside the `docs` subfolder of the `scVI` folder. Then, you can access the documentation in the HTML file at `docs/build/index.html`. 

## Testing 

Runtests can be executed via 

```
Pkg.test("scVI")
```

This checks whether basic functionality works: loads the PBMC dataset, initialises a `scVAE` model and start training (just 2 epochs for the sake of checking and faster runtime). Further tests to be added. 

------------
## TODO 

- [x] fix data loading --> from GoogleDrive
- [ ] document `loss_registry` field and option during training!
- [ ] deploy docs 
- [ ] integration of [`Muon.jl`](https://scverse.org/Muon.jl/dev/objects/) for data handling 
- [ ] support `gene_batch` and `gene_label` dispersion 

Contributions, reporting of bugs and unexpected behaviour, missing functionalities, etc. are all very welcome, please do get in touch!