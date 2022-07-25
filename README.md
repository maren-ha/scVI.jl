# README

![](logo/scvi-julia-logo.jpg)

A Julia package for fitting VAEs to single-cell data using count distributions. 
Based on the Python implementation in the [scvi-tools](https://github.com/scverse/scvi-tools) package. 

The scVI model was first proposed in [Lopez R, Regier J, Cole MB *et al.* Deep generative modeling for single-cell transcriptomics. *Nat Methods* **15**, 1053-1058 (2018)](https://doi.org/10.1038/s41592-018-0229-2). 

More on the much more extensive Python package ecosystem scvi-tools can be found on the 
[website](https://scvi-tools.org) and in the corresponding paper [Gayoso A, Lopez R, Xing G. *et al.* A Python library for probabilistic analysis of single-cell omics data. *Nat Biotechnol* **40**, 163–166 (2022)](https://doi.org/10.1038/s41587-021-01206-w). 

This is the documentation for the Julia version implementing basic functionality, including the following (non-exhausive list): 

- standard and linearly decoded VAE models 
- support for negative binomial generative distribution with and without zero-inflation 
- different ways of specifying the dispersion parameter 
- store data in a (very basic) Julia version of the Python [`AnnData`](https://anndata.readthedocs.io/en/latest/) objects 
- several built-in datasets 
- training routines supporting a wide range of customisable hyperparameters

## Docs 

Only available locally as of now: You can run `julia make.jl` inside the `docs` subfolder of the `scVI` folder. Then, you can access the current version of the documentation in the HTML file at `docs/build/index.html`. 

Hosting on Github with GithubActions/TravisCI is currently work in progress. 

## TODO 

*for workshop*
- [x] add PBMC8k data 
- [x] add linearly decoded VAE functionality 
- [x] add docs with `Documenter.jl`
- [x] add other datasets from scVI repo (https://github.com/scverse/scvi-tools/tree/master/scvi/data/_built_in_data)
- [x] add cortex data from download link 
- [x] implement highly variable gene filtering accoding to scanpy/Seuratv3 function 
- [x] add checks to data loading (dimensions etc. )
- [x] add in Tasic data + download script (?)
- [x] fix `init_library_size` function 
- [x] actually support more than one layer! 
- [x] support Poisson likelihood 
- [x] add docstrings (26/26)
- [ ] add supervised AE functionality 
-------
*for later*
- [ ] repo with data and preprocessing script for Tasic publically available 
- [ ] support gene_batch and gene_label dispersion 
- [ ] think about data loading (separate package? etc.)

## To test 

```
"""
Pkg.add("PkgTemplates")
using PkgTemplates
scvi_template = Template(; 
    user="maren-ha",
    authors="Maren Hackenberg",
    julia=v"1.7",
    plugins=[
        License(; name="MIT"),
    ],
)
scvi_template("scVI")
using Pkg;
Pkg.activate("scVI")
Pkg.add(["Random", "Flux", "Distributions", "SpecialFunctions", "ProgressMeter", "DataFrames", "CSV", "DelimitedFiles", "Loess", "LinearAlgebra", "HDF5", "VegaLite", "UMAP"])
"""
using Pkg;
Pkg.develop(path=string(@__DIR__))
#Pkg.activate("scVI")
using scVI 
Pkg.test("scVI")

# load cortex and try HVG selection 
adata = load_cortex(@__DIR__) # (or init_cortex_from_url())
hvgdict = highly_variable_genes(adata, n_top_genes=1200)
highly_variable_genes!(adata, n_top_genes=1200)
subset_to_hvg!(adata, n_top_genes=1200)
library_log_means, library_log_vars = init_library_size(adata)

m = scVAE(size(adata.countmatrix,2);
        library_log_means=library_log_means,
)
training_args = TrainingArgs(
    max_epochs=50, # 50 for 10-dim 
    weight_decay=Float32(1e-6),
)
train_model!(m, adata, training_args)
plot_umap_on_latent(m, adata; save_plot=true)

# Tasic data 
adata = load_tasic(joinpath(@__DIR__,"../data/"))
subset_tasic!(adata)

# load cortex and train model 
adata = load_cortex("scVI/data/")
n_batch = adata.summary_stats["n_batch"]
library_log_means, library_log_vars = init_library_size(adata) 

m = scVAE(size(adata.countmatrix,2);
        n_batch=n_batch,
        library_log_means=library_log_means,
)
print(summary(m))

training_args = TrainingArgs(
    max_epochs=50, # 50 for 10-dim 
    weight_decay=Float32(1e-6),
)
train_model!(m, adata, training_args)
plot_umap_on_latent(m, adata; save_plot=true)

# LDVAE 
m = scLDVAE(size(adata.countmatrix,2);
        n_batch=n_batch,
        library_log_means=library_log_means,
        use_activation=:encoder,
)
print(summary(m))
training_args = TrainingArgs(
    max_epochs=50, # 50 for 10-dim 
    weight_decay=Float32(1e-6),
)
train_model!(m, adata, training_args)
plot_umap_on_latent(m, adata; save_plot=true)

# PBMC dataset 
adata = load_pbmc("scVI/data/")
library_log_means, library_log_vars = init_library_size(adata) 
m = scVAE(size(adata.countmatrix,2);
        library_log_means=library_log_means,
        n_latent=2,
        dispersion=:gene_cell,
        gene_likelihood=:zinb
);
print(summary(m))

training_args = TrainingArgs(
    max_epochs=50, 
    lr = 1e-4,
    weight_decay=Float32(1e-6),
)
train_model!(m, adata, training_args)
register_latent_representation!(adata, m)
using VegaLite
@vlplot(:point, x=adata.scVI_latent[1,:], y=adata.scVI_latent[2,:], color=adata.celltypes, title="scVI latent representation")
```