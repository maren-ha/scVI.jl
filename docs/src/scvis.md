# The scvis model 

scVAE model with additional tSNE loss term. 
scVAE model with a linear decoder. The implementation is based on the [`scvi-tools` linearly decoded VAE](https://github.com/scverse/scvi-tools/blob/b33b42a04403842591c04e414c8bb4099eaf7006/scvi/model/_linear_scvi.py#L21). According to the `scvi-tools` authors, this is turn is based on the model proposed in [Svensson et al, 2020](https://academic.oup.com/bioinformatics/article/36/11/3418/5807606).

Ding, J., Condon, A. & Shah, S.P. Interpretable dimensionality reduction of single cell transcriptome data with deep generative models. Nat Commun 9, 2002 (2018). https://doi.org/10.1038/s41467-018-04368-5


scvis is a statistical model that captures and visualizes the low-dimensional structures in single-cell gene expression data. It is robust to the number of data points and learns a probabilistic parametric mapping function to add new data points to an existing embedding.

The article also shows how scvis is used to analyze four single-cell RNA-sequencing datasets to create interpretable two-dimensional representations of the high-dimensional data.