---
layout:     post   				    # 使用的布局（不需要改）
title:      GiniClust				# 标题 
subtitle:   单细胞rare cell鉴定分析 #副标题
date:       2020-06-23				# 时间
author:     CHY					# 作者
header-img: img/wallhaven-2067.png 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - 单细胞
---
GiniClust3是专门开发用于rare cell type鉴定的软件，它主要通过Gini index来鉴定与rare cell type相关的基因。基于a cluster-aware, weighted consensus clustering approach，他将Gini index和Fano factor的结果进行整合来鉴定rare cell type.
![GiniClust3](https://github.com/chenhongyubio/chenhongyubio.github.io/raw/master/img/GiniCluster3.png)

```
import scanpy as sc
import numpy as np
import giniclust3 as gc
import anndata

adataRaw=sc.read_csv("/public/home/fanlj/chenhy/ara_Seurat_result/all/RData/RData60_2000/50pc/res0.6/integrated_count.csv",first_column_names=True)

# 数据过滤
# sc.pp.filter_cells(adataRaw,min_genes=3)
# sc.pp.filter_genes(adataRaw,min_cells=200)

# 格式标准化
###example csv file is col:cells X row:genes. Skip this step if the input matrix is col:genes X row:cells
# 行为细胞，列为基因
adataSC=anndata.AnnData(X=adataRaw.X.T,obs=adataRaw.var,var=adataRaw.obs)

# 数据标准化
# sc.pp.normalize_per_cell(adataSC, counts_per_cell_after=1e4)

# GiniCluster
gc.gini.calGini(adataSC) ###Calculate Gini Index
adataGini=gc.gini.clusterGini(adataSC,neighbors=3) ###Cluster based on Gini Index

# FanoFactorClust
gc.fano.calFano(adataSC) ###Calculate Fano factor
adataFano=gc.fano.clusterFano(adataSC) ###Cluster based on Fano factor

# 聚类整合
consensusCluster={}
consensusCluster['giniCluster']=np.array(adataSC.obs['rare'].values.tolist())
consensusCluster['fanoCluster']=np.array(adataSC.obs['fano'].values.tolist())
gc.consensus.generateMtilde(consensusCluster) ###Generate consensus matrix
gc.consensus.clusterMtilde(consensusCluster) ###Cluster consensus matrix
# np.savetxt("final.txt",consensusCluster['finalCluster'], delimiter="\t",fmt='%s')

# UMAP
adataGini.obs['final']=consensusCluster['finalCluster']
adataFano.obs['final']=consensusCluster['finalCluster']
png("/public/home/fanlj/chenhy2/rare_cell/result/Gini_plot.png")
gc.plot.plotGini(adataGini)
dev.off()
png("/public/home/fanlj/chenhy2/rare_cell/result/Fano_plot.png")
gc.plot.plotFano(adataFano)
dev.off()
```

#### Gini index计算
基尼指数越小，则数据集纯度越高。基尼指数偏向于特征值较多的特征，类似信息增益。基尼指数可以用来度量任何不均匀分布，是介于 0~1 之间的数，0 是完全相等，1 是完全不相等。
![Gini index](https://github.com/chenhongyubio/chenhongyubio.github.io/raw/master/img/Gini_index.png)
当较少一部分细胞的表达量很高时，A区域的面积也会增大。

#### FanoFactor
Fano因子定义为每个基因平均表达值的方差。

#### 聚类结果融合
cluster-aware, weighted ensemble approach<br>
在GiniClust分群中，更高的权重分配给rare cell clusters,在Fano factor为基础的k-means聚类则相反。<br>
