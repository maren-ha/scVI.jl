import os as os
from multiprocessing.sharedctypes import Value
import scvi
import anndata as ad
import scipy as sp
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
import gc
import multiprocessing
from tqdm import tqdm
from timeit import default_timer as timer
os.environ['R_HOME'] = '/Library/Frameworks/R.framework/Resources'
import scib

# read cite-seq data
adata_protein = sc.read("data_sampled/adata_cite_protein_subsample_5000_cells_rep_0.h5ad")
adata_protein.var_names_make_unique()

adata_rna =  sc.read("data_sampled/adata_cite_gex_subsample_5000_cells_rep_0.h5ad")
adata_rna.var_names_make_unique()

# Search for common barcodes (cells)
common_barcodes = list(set(adata_rna.obs_names).intersection(adata_protein.obs_names))
adata_rna_sub = adata_rna[common_barcodes].copy()
adata_protein_sub = adata_protein[common_barcodes].copy()
#size_factors phase             cell_type  pseudotime_order_GEX batch  pseudotime_order_ADT  is_train
# Concatenate data sets 
combined_dat = ad.concat([adata_rna_sub,adata_protein_sub], axis=1)
combined_dat.obs["cell_type"] = adata_rna_sub.obs["cell_type"]
combined_dat.obs["batch"] = adata_rna_sub.obs["batch"]
combined_dat.obs["pseudotime_order_GEX"] = adata_rna_sub.obs["pseudotime_order_GEX"]
combined_dat.obs["pseudotime_order_ADT"] = adata_rna_sub.obs["pseudotime_order_ADT"]
combined_dat.obs["phase"] = adata_rna_sub.obs["phase"]
combined_dat.obs["modality"] = pd.Series(("paired",) * adata_rna_sub.shape[0]).values
combined_dat.obsm["X_pca"] = adata_rna_sub.obsm["X_pca"]
combined_dat.obsm["X_umap"] = adata_rna_sub.obsm["X_umap"]
combined_dat.var["modality"] = pd.concat([pd.Series(("Gene Expression",) * adata_rna_sub.shape[1]),pd.Series(("Proteins",) * adata_protein_sub.shape[1])], axis=0).values
combined_dat.obsm["protein_expression"] = adata_protein_sub.layers['counts'].toarray()

# Sort features 
combined_dat = combined_dat[:, combined_dat.var["feature_types"].argsort()].copy()
combined_dat.var
combined_dat.var_names_make_unique()

########################
### Model evaluation ###
########################

# Copy data to new object
combined_dat_int = combined_dat.copy()

# Load embedding
latent_representation = "./experiment_08_07_2022_0950/lat_mean_train_mean.csv"            
embedding = pd.read_csv(latent_representation, header=None)
embedding = embedding.transpose()
embedding = embedding[[0,1,2,3,4,5,6,7,8,9]]
embedding.index = embedding.index.str.replace(r'CITE~|Multiome~', '')
embedding = embedding.loc[combined_dat_int.obs_names, :]
embedding = embedding.to_numpy()

# Copy embedding to integrated object
combined_dat_int.obsm['X_emb'] = embedding.copy()
label_key = 'cell_type'
batch_key = 'batch'
sc.pp.neighbors(combined_dat)
sc.tl.leiden(combined_dat)
sc.pp.neighbors(combined_dat_int,use_rep='X_emb')
sc.tl.leiden(combined_dat_int)

# Caluclate evaluation metrics
# Set True/False depending on which metrics we need 
# calculate metrics with metrics from scib 
eval = scib.metrics.metrics(combined_dat, combined_dat_int, batch_key=batch_key, label_key=label_key, isolated_labels_asw_=True,silhouette_=True,hvg_score_=False,pcr_=True,isolated_labels_f1_=True,nmi_=True,ari_=True,graph_conn_=True,embed = 'X_emb')

trajectory_score = scib.metrics.trajectory_conservation(
                combined_dat,
                combined_dat_int,
                label_key=label_key,
                batch_key = batch_key,
                pseudotime_key = 'pseudotime_order_GEX'
)

cell_cycle = scib.metrics.cell_cycle(
                combined_dat, 
                combined_dat_int, 
                batch_key,
                embed='X_emb',
                organism='human',
                n_comps=50,
)

ASW_site = scib.metrics.metrics(combined_dat, combined_dat_int, batch_key=batch_key, label_key=label_key, 
                silhouette_=True,
                embed = 'X_emb'
)

# Transpose and add information (model, cells, repition)
eval = eval.transpose()
eval["trajectory"] = trajectory_score
eval["cell_cycle_conservation"] = cell_cycle
eval["ASW_label/site"] = ASW_site.loc['ASW_label/batch']
                