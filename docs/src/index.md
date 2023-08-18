# scVI.jl

![](assets/scvi-julia-logo.jpg)

A Julia package for fitting VAEs to single-cell data using count distributions. 
Based on the Python implementation in the [`scvi-tools`](https://github.com/scverse/scvi-tools) package. 

## Overview

The scVI model was first proposed in [Lopez R, Regier J, Cole MB *et al.* Deep generative modeling for single-cell transcriptomics. *Nat Methods* **15**, 1053-1058 (2018)](https://doi.org/10.1038/s41592-018-0229-2). 

More on the much more extensive Python package ecosystem [`scvi-tools`](https://github.com/scverse/scvi-tools) can be found on the 
[website](https://scvi-tools.org) and in the corresponding paper [Gayoso A, Lopez R, Xing G. *et al.* A Python library for probabilistic analysis of single-cell omics data. *Nat Biotechnol* **40**, 163–166 (2022)](https://doi.org/10.1038/s41587-021-01206-w). 

This is the documentation for the Julia version implementing basic functionality, including: 

- standard and linearly decoded VAE models 
- support for negative binomial generative distribution w/o zero-inflation, Poisson distribution, Gaussian and Bernoulli distribution
- different ways of specifying the dispersion parameter 
- library size encoding 
- representing data as a Julia `AnnData` object based on [Muon.jl](https://scverse.org/Muon.jl/dev/) fully analogous to Python's [anndata](https://anndata.readthedocs.io/en/latest/) object + standard slicing and subsetting operations
- preprocessing functions operating directly on the `AnnData` object analogous to [scanpy](https://scanpy.readthedocs.io/en/stable/) functions: filtering, highly variable gene seletion, transformations, dimension reduction, etc. 
- several built-in datasets (see below)
- training routines supporting a wide range of customizable hyperparameters including a freely definable layer structure
- easily customizable loss functions for shaping the latent space, e.g., to resemble a t-SNE or UMAP embedding

## Installation

The package can be downloaded from the [Github repo](https://github.com/maren-ha/scVI.jl) and added with the Julia package manager via 

```
julia> ]
pkg > add "https://github.com/maren-ha/scVI.jl"
```

or alternatively by 

```
julia> using Pkg; Pkg.add(url="https://github.com/maren-ha/scVI.jl")
```


## Contents

```@contents 
Pages = [
    "DataProcessing.md", 
    "scVAE.md",
    "scLDVAE.md", 
    "ModelFunctions.md", 
    "Training.md", 
    "Evaluation.md", 
    "Utils.md"
]
```