'''
RankGenes
'''
import numpy as np
import pandas as pd

def RankGenes(adata, layer='SaverX', cutoff=1000, specific=1):
    tfidf = layer+"_TFIDF"
    if tfidf not in adata.varm:
        raise ValueError(f".varm['{tfidf}'] was not found. Run `CalculateTFIDF` with `layer='{layer}'` first")
    rankgenes = adata.varm[tfidf].rank(axis=0,method='average',
                                       numeric_only=None,na_option='keep',
                                       ascending=False,pct=False)
    ALL = rankgenes.columns.to_list()
    MarkerList = pd.DataFrame(columns=ALL, dtype=bool)
    for region in ALL:
        cutrankgenes = rankgenes.sort_values(by=region) < cutoff
        rankNgenes = cutrankgenes[np.sum(cutrankgenes, 1) == specific]
        MarkerList = pd.concat([MarkerList,rankNgenes[rankNgenes[region]]],axis=0)
    adata.uns['MarkerList'] = MarkerList
    print(pd.DataFrame(MarkerList.sum(0),columns=["Number of markers"]))
    print('Total number:{}'.format(MarkerList.sum().sum()))
    return adata