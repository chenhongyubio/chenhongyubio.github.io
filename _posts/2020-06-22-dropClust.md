---
layout:     post   				    # 使用的布局（不需要改）
title:      dropClust				# 标题 
subtitle:   大规模单细胞数据聚类分析 #副标题
date:       2020-06-22				# 时间
author:     CHY					# 作者
header-img: img/wallhaven-2066.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - 单细胞
---
dropclust方法是用于针对大规模单细胞数据进行聚类分析的算法，采用局部敏感哈希（LSH）的方法去找到细胞的Nearest Neighbour（加速聚类的过程）。然后使用这些Neighbour的信息，提出了一种细胞采样的方法Structure Preserving Sampling (SPS) ，使得细胞数量较少的那些类别被采样到的比率会相对比较大，降低由于采样而导致的聚类精度损失。提出的整个过程能够处理大规模、高稀疏度的单细胞RNA数据。此外dropClust还可用于整合分析。<br>
dropclust在线版链接：https://debsinha.shinyapps.io/dropClust/ <br>

#### R包安装
```
library(devtools)
install_github("debsin/dropClust", dependencies = T)
library(dropClust)
set.seed(0)
```

#### 数据加载
load UMI count expression data from three input files.
* count matrix file in sparse format
* transcriptome identifiers as a TSV file and
* gene identifiers as a TSV file
```
sce <-readfiles(path = "C:/Projects/dropClust/data/pbmc3k/hg19/")
```

#### 数据预处理
remove poor quality cells and genes，还能除去一定的批次效应。
```
sce<-FilterCells(sce)
sce<-FilterGenes(sce)
```

#### 数据标准化
```
sce <- CountNormalize(sce)
```

#### 选择HVG
基于分散系数dispersion index选择基因。
```
sce<-RankGenes(sce, ngenes_keep = 1000)
```

#### Structure Preserving Sampling保存结构抽样
```
sce <- Sampling(sce)
```

#### 基于PCA结果再次选择基因
```
sce<-RankPCAGenes(sce)
```

#### 聚类
默认情况，会返回Louvain认为合适的类群会返回，也可以自己设定类群数，未被选择的细胞根据conf参数来进行分配。
```
sce <- Cluster(sce, method = "default", conf = 0.8)
```

#### 数据可视化
```
sce<-PlotEmbedding(sce, embedding = "umap", spread = 10, min_dist = 0.1)
plot_data = data.frame("Y1" = reducedDim(sce,"umap")[,1], Y2 = reducedDim(sce, "umap")[,2], color = sce$ClusterIDs)
ScatterPlot(plot_data,title = "Clusters")

# 特异表达基因
DE_genes_all = FindMarkers(sce, selected_clusters=NA, lfc_th = 1, q_th =0.001, nDE=30)
write.csv(DE_genes_all$genes, 
          file = file.path(tempdir(),"ct_genes.csv"),
          quote = FALSE)
marker_genes = c("S100A8", "GNLY", "PF4")
p<-PlotMarkers(sce, marker_genes) 
p<-PlotHeatmap(sce, DE_res = DE_genes_all$DE_res,nDE = 10)
print(p)     
```