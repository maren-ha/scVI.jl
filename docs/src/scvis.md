# The scvis model 

scVAE model with additional tSNE loss term, based on the scvis model proposed in [Ding J, Condon A and Shah SP Interpretable dimensionality reduction of single cell transcriptome data with deep generative models. *Nat Commun* **9**, 2002 (2018)](https://doi.org/10.1038/s41467-018-04368-5).

From the abstract: 
> scvis is a statistical model that captures and visualizes the low-dimensional structures in single-cell gene expression data. It is robust to the number of data points and learns a probabilistic parametric mapping function to add new data points to an existing embedding.

## The additional t-SNE loss component

The model is based on adding a t-SNE objective to the VAE loss function (i.e., the ELBO). This component is supposed to help structure the latent representation to look more like t-SNE. 

Similar to the usual t-SNE objective, a matrix transition of transition probabilities has to be calculated. For this, individual perplexities have to be calculated. 


Hbeta!

perplexities

Based on these transition probabilities, the t-SNE loss component is calculated and integrated with the standard ELBO. 
