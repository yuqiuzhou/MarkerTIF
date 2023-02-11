'''
Calculate TFIDF.
'''
import numpy as np
import pandas as pd

import tensorflow as tfw
import tensorflow.experimental.numpy as tnp
tnp.experimental_enable_numpy_behavior()
tfw.compat.v1.disable_eager_execution()
tfw.get_logger().setLevel('ERROR')

from skimage import measure
from skimage import morphology as sm
import warnings
warnings.filterwarnings("ignore")

def CalculateTFIDF(adata,layer="SaverX",
                   label="RegionLabel",
                   omega_s=1,alpha=4,
                   min_exp=0.1,threads=0,
                   LOWER_BOUND=1e-20):
    label = label
    layer = layer
    adata.X = adata.layers[layer].copy()
    ## Create Region Label Mask
    adata_temp = adata[:,np.any(adata.X,0) > min_exp]  # temporary adata for calculation
    adata_temp = CreateMask(adata_temp, scale_factor=1)
    ## Extract Skeleton of Region
    adata_temp = ExtractSkeleton(adata_temp, label_key=label)
    Mask = adata_temp.uns['Mask'].copy()
    Nrow, Ncol = Mask.shape
    NS = Nrow*Ncol
    NG = adata_temp.shape[1]

    PGDataFrame = np.zeros(shape=(NG, NS))
    PGDataFrame[:,adata_temp.obs['spotID']] = adata_temp.X.T
    RegionLabelDF = np.full(NS, "", dtype = "object")
    RegionLabelDF[adata_temp.obs['spotID']] = adata_temp.obs[label]
    RegionLabelDF = pd.DataFrame(RegionLabelDF,columns = [label])
    RegionLabelMask = pd.get_dummies(RegionLabelDF[label])
    RegionLabelMask = RegionLabelMask.loc[:,RegionLabelMask.columns != ""]

    SkeletonMask = np.zeros(NS, dtype='int')
    isSkeleton = 1*(adata_temp.obs[label+'_Skeleton'] == adata_temp.obs[label+'_Skeleton'])
    SkeletonMask[adata_temp.obs['spotID']] = isSkeleton
    SkeletonWeight = np.array(omega_s * SkeletonMask + (1-omega_s) * (1-SkeletonMask))
    SkeletonWeight = SkeletonWeight.reshape(NS,1)

    ## Modified TF-IDF
    tf = tfw.compat.v1
    tf.disable_v2_behavior()
    tf.reset_default_graph()

    NR = RegionLabelMask.shape[1]

    ########Create Masked PG Image#########
    ## Data placeholders
    SkeletonWeight_ = tf.placeholder(tf.float64, shape = (NS,1), name = "SkeletonWeight")
    RegionLabelMask_ = tf.placeholder(tf.float64, shape = (NS,NR), name = "RegionLabelMask")
    RegionLabelMask_tensor_list = list()
    for pg in range(NG):
        RegionLabelMask_tensor_list.insert(pg,RegionLabelMask_)
    RegionLabelMask__ = tf.stack(RegionLabelMask_tensor_list, axis = 0)

    RegionLabelMaskImage_ = tf.transpose(tf.reshape(RegionLabelMask_,(Nrow,Ncol,NR)),(2,0,1))

    WeightedRegionLabelMask_ = tf.multiply(SkeletonWeight_,RegionLabelMask_)
    
    PGDataFrame_ = tf.placeholder(tf.float64, shape = (NG,NS), name = "PGDataFrame")
    PGDataFrame_cut_ = tf.where(PGDataFrame_ < min_exp, tf.zeros_like(PGDataFrame_), PGDataFrame_)

    #PGImage_ = tf.reshape(PGDataFrame_,(NG,Nrow,Ncol))
    #PGImage_cut_ = tf.reshape(PGDataFrame_cut_,(NG,Nrow,Ncol))

    PGDataFrame__ = tf.reshape(PGDataFrame_,(NG,NS,1))
    PGDataFrame__cut_ = tf.reshape(PGDataFrame_cut_,(NG,NS,1))

    maskPGDataFrame_ = tf.multiply(PGDataFrame__, RegionLabelMask__) 
    maskPGDataFrame_cut_ = tf.multiply(PGDataFrame__cut_, RegionLabelMask__) 

    #maskPGImage_ = tf.reshape(maskPGDataFrame_,(NG,Nrow,Ncol,NR))
    maskPGImage_cut_ = tf.reshape(maskPGDataFrame_cut_,(NG,Nrow,Ncol,NR))

    ## Start the graph and inference
    threads = int(threads)

    session_conf = tf.ConfigProto(intra_op_parallelism_threads = threads,
                                    inter_op_parallelism_threads = threads)
    sess = tf.Session(config = session_conf)
    fd_full = {"PGDataFrame:0": PGDataFrame, "RegionLabelMask:0": RegionLabelMask,"SkeletonWeight:0": SkeletonWeight}
    [maskPGImage_cut,RegionLabelMaskImage] = sess.run([maskPGImage_cut_,RegionLabelMaskImage_], feed_dict = fd_full)
    sess.close()

    labelsPGImage_cut = np.zeros(maskPGImage_cut.shape)
    labelsRegionLabelMaskImage = np.zeros(RegionLabelMaskImage.shape,dtype = int)

    NCgr = np.zeros((maskPGImage_cut.shape[0],maskPGImage_cut.shape[3]))
    NPr = np.zeros((1,NR))

    for idx in range(NG):
        image = maskPGImage_cut[idx]
        img = 1*(image > min_exp)
        for i in range(NR):
            labelsPGImage_cut[idx,:,:,i] = measure.label(img[:,:,i],connectivity=2)
            NCgr[idx,i] = labelsPGImage_cut[idx,:,:,i].max()

    for i in range(NR):
        labelsRegionLabelMaskImage[i] = measure.label(RegionLabelMaskImage[i],connectivity=2)
        NPr[0,i] = int(labelsRegionLabelMaskImage[i].max())

    ######TF-IDF
    ##Data placeholders
    NCgr_ = tf.placeholder(tf.float64, shape = (NG,NR), name = "NCgr")
    NPr_ = tf.placeholder(tf.float64, shape = (1,NR), name = "NPr")

    ###Fraction of Coverage as weight ρ
    A_gr_ = tf.reduce_sum(tf.cast(maskPGDataFrame_ > min_exp,dtype=tf.float64),1)
    NSr_ = tf.reshape(tf.reduce_sum(RegionLabelMask_,0),(1,NR))
    temp1 = NCgr_ - NPr_
    temp2 = tf.where(temp1 < 0, tf.ones_like(temp1), temp1)
    temp2_01_ = (temp2 - tf.reduce_min(temp2)) / (tf.reduce_max(temp2) - tf.reduce_min(temp2))
    eta_ = tf.exp(temp2_01_)
    p_gr_ = tf.divide(A_gr_ ,tf.multiply(eta_,NSr_))
    p_gr_ = tnp.round(p_gr_, 3)
    ###Frequency of Expression
    E_gr_ = tf.matmul(PGDataFrame_,WeightedRegionLabelMask_)
    #E_gr_ = tf.reduce_sum(maskPGDataFrame_,1)
    f_gr_ = tf.divide(E_gr_,tf.reshape(tf.reduce_sum(WeightedRegionLabelMask_,0),(1,NR)))
    f_gr_ = tnp.round(f_gr_, 3)

    ##TF
    tf_gr_ = tf.multiply(p_gr_, f_gr_)

    ##IDF
    p_g_ = tf.reduce_sum(p_gr_,1)
    p_g01_ = 7*(p_g_ - tf.reduce_min(p_g_)) / (tf.reduce_max(p_g_) - tf.reduce_min(p_g_)) + LOWER_BOUND
    idf_g_ = tf.multiply(tf.constant(alpha,dtype=tf.float64),tf.divide(tf.log((NR*NR)/p_g01_),tf.log(tf.constant(NR,dtype=tf.float64))))
    idf_g_ = tf.reshape(idf_g_,(NG,1))
    tfidf_gr_ = tf.multiply(tf_gr_,idf_g_)
    tfidf_gr_ = tnp.round(tfidf_gr_, 2)
    threads = int(threads)

    session_conf = tf.ConfigProto(intra_op_parallelism_threads = threads,
                                    inter_op_parallelism_threads = threads)
    sess = tf.Session(config = session_conf)
    fd_full = {"PGDataFrame:0": PGDataFrame, "RegionLabelMask:0": RegionLabelMask, 
               "NCgr:0": NCgr,"NPr:0": NPr,"SkeletonWeight:0": SkeletonWeight}         
    [TFIDFgr,tf_gr,idf_g,p_g] = sess.run([tfidf_gr_,tf_gr_,idf_g_,p_g_], feed_dict = fd_full)
    sess.close()

    TFIDFgr = pd.DataFrame(TFIDFgr,index = adata_temp.var.index,columns = RegionLabelMask.columns,dtype = np.float32)
    adata_temp.varm["{}_TFIDF".format(layer)] = TFIDFgr
    return adata_temp


# Create a mask image from the spatial coordinates.
def CreateMask(adata, scale_factor=1):
    adata_copy = adata.copy()
    A = 'spatial' in adata_copy.uns.keys()
    B = 'array_col' in adata_copy.obs.keys()
    C = 'array_row' in adata_copy.obs.keys()
    if A and B and C:
        coordinates = adata_copy.obs[['array_row', 'array_col']]
    else:
        coordinates = pd.DataFrame(adata_copy.obsm['spatial'], index=adata_copy.obs.index,
                                   columns=['array_col', 'array_row'])    

    if (coordinates.max()>1000).any():
        coordinates = (coordinates/scale_factor).astype(int)

    evencol = coordinates[coordinates['array_col'] % 2 == 0]['array_row']
    oodcol = coordinates[coordinates['array_col'] % 2 != 0]['array_row']
    aligned = len(set(evencol).intersection(oodcol)) != 0
    coordinates = coordinates - coordinates.min()

    if aligned:
        nrow = coordinates['array_row'].max() + 1 
        ncol = coordinates['array_col'].max() + 1
        Mask = np.zeros((nrow, ncol))
        spotID = []
        for i in range(coordinates.shape[0]):   
            Mask[coordinates['array_row'][i], coordinates['array_col'][i]] = 1
            spotID.append(coordinates['array_row'][i]*ncol+coordinates['array_col'][i])
    else:
        nrow = coordinates['array_row'].max() + 1
        ncol = int((coordinates['array_col'].max() + 1)/2) + 1
        Mask = np.zeros((nrow, ncol))
        spotID = []
        for i in range(coordinates.shape[0]):   
            Mask[coordinates['array_row'][i], coordinates['array_col'][i] // 2] = 1
            spotID.append(coordinates['array_row'][i]*ncol+(coordinates['array_col'][i] // 2))
    
    adata_copy.uns['Mask'] = Mask
    coordinates['spotID'] = spotID
    coordinates = coordinates.loc[adata_copy.obs.index,:]
    adata_copy.obs[['array_col', 'array_row', 'spotID']] = coordinates
    return adata_copy

def ExtractSkeleton(adata, label_key='RegionLabel'):
    Regions = list(adata.obs[label_key].cat.categories)
    Skeleton = pd.Series([None]*adata.shape[0], dtype="category")
    Skeleton.index = adata.obs.index
    Skeleton = Skeleton.cat.set_categories(Regions)
    RegionOneHot = pd.get_dummies(adata.obs[label_key])
    Mask = adata.uns['Mask']
    for r in Regions:
        bg = np.zeros(Mask.shape).flatten()
        bg[adata.obs['spotID']] = RegionOneHot[r]
        img = bg.reshape(Mask.shape)*Mask
        img = sm.skeletonize(img)
        Skeleton[img.flatten()[adata.obs['spotID']]] = r
    adata.obs[label_key+'_Skeleton'] = Skeleton
    return adata