# README

![](logo/scvi-julia-logo.jpg)

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
- [ ] add supervised AE functionality 
- [ ] add docstrings (9/26)
-------
*for later*
- [ ] repo with data and preprocessing script for Tasic publically available 
- [ ] support gene_batch and gene_label dispersion 
- [ ] think about data loading (separate package? etc.)

## Docs 

(only works locally, not deployed yet)
[docs](docs/build/index.html)

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