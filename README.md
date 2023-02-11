# MarkerTIF
A Spatial Transcriptome Differential Analysis Algorithm and Its Application in Extracting Mouse Brain Structure-Specific Markers
## Import Modules

```python
import numpy as np
import scanpy as sc
# pip install markertif
import markertif
```

## Data input


```python
adata = sc.read_h5ad("~/data/SampleA.h5ad")
```


```python
adata
```




    AnnData object with n_obs × n_vars = 2686 × 14698
        obs: 'in_tissue', 'array_row', 'array_col', 'n_counts', 'SizeFactor', 'leiden', 'cluster', 'n_genes', 'spotID', 'cluster_Skeleton', 'hip'
        var: 'gene_ids', 'feature_types', 'genome', 'n_cells', 'mt', 'highly_variable', 'means', 'dispersions', 'dispersions_norm'
        uns: 'Mask', 'cluster_colors', 'defaultX', 'hip_colors', 'hvg', 'log1p', 'moranI', 'spatial', 'spatial_neighbors'
        obsm: 'spatial'
        varm: 'SaverX_TFIDF'
        layers: 'SaverX', 'SpatialX'
        obsp: 'spatial_connectivities', 'spatial_distances'



##### MarkerTIF takes Anndata as the processing object. It is suggested to calculate the gene expression matrix after noise reduction or standardization, and store it in "adata.uns['layers']" with user-defined key names. Following the convention of 10x Visium, the spatial coordinates are stored in "adata.obs['array_row']" and "adata.obs['array_col']"

## CalculateTFIDF


```python
# Specify the layer that will be used for the calculation, and the label that will be used for the difference analysis
adata = markertif.CalculateTFIDF(adata,layer="SaverX",label="cluster")
```

    2023-02-11 19:56:04.542107: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2023-02-11 19:56:04.548676: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
    2023-02-11 19:56:04.548724: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
    2023-02-11 19:56:04.548746: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (fat01): /proc/driver/nvidia/version does not exist

## RankGenes

```python
# cutoff: Ranking threshold, default 1000
adata = markertif.RankGenes(adata, layer='SaverX', cutoff=1000)
```

                                   Number of markers
    Cortex_1                                      15
    Cortex_2                                      17
    Cortex_3                                      42
    Cortex_4                                      96
    Cortex_5                                      43
    Fiber_tract                                  576
    Hippocampus                                   45
    Hypothalamus_1                               161
    Hypothalamus_2                                27
    Lateral_ventricle                            240
    Pyramidal_layer                              261
    Pyramidal_layer_dentate_gyrus                 70
    Striatum                                      50
    Thalamus_1                                   289
    Thalamus_2                                    31
    Total number:1963


##### The results are stored in the adata.uns['MarkerList']

## Visualization


```python
tfidfmarker = adata.uns['MarkerList'][adata.uns['MarkerList']['Fiber_tract']].head(3)
```


```python
sc.pl.spatial(adata, size=1.5, layer='SaverX', color=tfidfmarker.index)
```




```python
tfidfmarker = adata.uns['MarkerList'][adata.uns['MarkerList']['Pyramidal_layer_dentate_gyrus']].head(3)
```


```python
sc.pl.spatial(adata, size=1.5, layer='SaverX', color=tfidfmarker.index)
```
