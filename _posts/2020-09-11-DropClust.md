---
layout: post # 使用的布局（不需要改）
title: DropClust # 标题
subtitle: 针对大型单细胞数据分析 #副标题
date: 2020-09-15 # 时间
author: CHY # 作者
header-img: img/wallhaven-83lyqk.jpg #这篇文章标题背景图片
catalog: true # 是否归档
tags: #标签
  - 单细胞
---

#### Motivation 想要解决的问题

1. 基于 Nearest Neighbour Network 的聚类方法，由于涉及到对细胞与细胞两两之间相似度的计算（即把细胞看成是向量，计算向量之间的相似度），往往会造成很大的计算开销。
2. 对细胞进行简单的 Random Sampling 会使得细胞数量较少的那些类别有可能无法被采样，导致最终聚类的精度不够高。（Highly Parallel Genome-Wide Expression Profling of Individual Cells Using Nanoliter Droplets ）
3. 基于 KMeans 的方法有两个明显的缺点。一是需要明确指定聚类的数量，这对发现细胞类别来说是不足的；二是难以挖掘那些非球状的聚类。

#### Contribution 解决办法

1. 采用局部敏感哈希（LSH）的方法去找到细胞的 Nearest Neighbour。（加速聚类的过程）
2. 使用这些 Neighbour 的信息，提出了一种细胞采样的方法 Structure Preserving Sampling (SPS) ，使得细胞数量较少的那些类别被采样到的比率会相对比较大，降低由于采样而导致的聚类精度损失。

### 分析流程

输入表达矩阵：行为细胞，列为基因

#### 数据预处理

1. 基因过滤：选择那些至少在三个 Cell 下，UMI 计数大于等于 3 的基因
2. UMI Normalization。对于每一行（Cell），其中的每个值先除以这一行所有 UMI 值的总和，再乘上这一行所有 UMI 值的中位数。
3. 选择变异较大的基因，即变异系数 = 方差除以平均值
4. log2 转换

#### 矩阵筛选

通过 Structure Preserving Sampling 和 Gene Selection 的方法，筛选出一个小规模的表达矩阵。<br>
Structure Preserving Sampling：

1. 首先在原始的基因表达矩阵中，筛选出至少 20000 个或 1/3 的 Cell 出来，组成一个新的表达矩阵。然后通过 LSHForest 算法构建一个 Nearest Neighbor Network，最后通过社区发现算法 Louvain 完成对这个新的表达矩阵中的 Cell 的聚类。
2. 在已经被分类好的这些 Cell 当中通过一定的规则进行第二次采样。
3. 对这个新的表达矩阵采用 PCA 算法，得到 top 50 个特征向量（这里是对矩阵的列维度进行降维）。然后对这些特征向量使用高斯混合模型算法，对于每个特征向量，就能得到被称为 Gaussian Component 的这些值。

#### 层次聚类

通过一个简单的层次聚类方法对这个小的表达矩阵的 Cell 进行分类。<br>
在层次聚类中，则直接使用欧式距离计算 Cell 之间两两的相似度。<br>

#### LSH 分配遗漏的细胞

通过 LSH 的方法把没有被采样到的那些 Cell 分配到已经分好类的这些 Cell 所属的类别中。<br>
首先对采样到的 Cell 聚类构建 LSHForest，然后将每一个未被采样到的 Cell 当作一个查询点，对其进行 LSH 查询找到 k 个 nearest neighbor（k 默认为 5）。然后，哪个聚类中的 nearest neighbor 的数目最多，这个 Cell 就属于哪个聚类。这样，就完成了对所有细胞的聚类过程。<br>

### 示例代码

```
# 常规分析流程
library(dropClust)
set.seed(0)

## 数据加载(10x格式数据)
sce <-readfiles(path = "F:\\资料用户共享\\OneDrive\\文章\\生物角度\\Dropclust\\pbmc3k_raw_gene_bc_matrices\\hg19\\")

## 预处理
sce<-FilterCells(sce)
sce<-FilterGenes(sce)

## 标准化
sce<-CountNormalize(sce)

## 选择高变基因
sce<-RankGenes(sce, ngenes_keep = 1000)

## SPS抽样
sce<-Sampling(sce)

## 基于PCA进行再次基因选择
sce<-RankPCAGenes(sce)

## 聚类
sce<-Cluster(sce, method = "default", conf = 0.8)

## 可视化
sce<-PlotEmbedding(sce, embedding = "umap", spread = 10, min_dist = 0.1)
plot_data = data.frame("Y1" = reducedDim(sce,"umap")[,1], Y2 = reducedDim(sce, "umap")[,2], color = sce$ClusterIDs)
ScatterPlot(plot_data,title = "Clusters")

## 差异基因分析
DE_genes_all = FindMarkers(sce, selected_clusters=NA, lfc_th = 1, q_th =0.001, nDE=30)
write.csv(DE_genes_all$genes,
          file = file.path(tempdir(),"ct_genes.csv"),
          quote = FALSE)

## Marker gene小提琴图
marker_genes = c("S100A8", "GNLY", "PF4")
p<-PlotMarkers(sce, marker_genes)

## 热图
p<-PlotHeatmap(sce, DE_res = DE_genes_all$DE_res,nDE = 10)
print(p)
```

```
# 数据整合分析

## 数据加载，构建list（三组数据）
## 每个数据必须是SingleCEllExperiment对象
library(dropClust)
load(url("https://raw.githubusercontent.com/LuyiTian/CellBench_data/master/data/sincell_with_class.RData"))
objects = list()
objects[[1]] = sce_sc_10x_qc
objects[[2]] = sce_sc_CELseq2_qc
objects[[3]] = sce_sc_Dropseq_qc

## merge data
all.objects = objects
merged_data<-Merge(all.objects)

## 校正降维
set.seed(1)
dc.corr <-  Correction(merged_data,  method="default", close_th = 0.1, cells_th = 0.1,
                       components = 10, n_neighbors = 30,  min_dist = 1)

## 聚类
dc.corr = Cluster(dc.corr,method = "kmeans",centers = 3)

## 可视化
ScatterPlot(dc.corr, title = "Clusters")

## 批次效应去除（fastmnn)
merged_data.fastmnn<-Merge(all.objects,use.de.genes = FALSE)
set.seed(1)
mnn.corr <-  Correction(merged_data.fastmnn,  method="fastmnn", d = 10)
mnn.corr = Cluster(mnn.corr,method = "kmeans",centers = 3)
ScatterPlot(mnn.corr, title = "Clusters")

## 差异基因分析
de<-FindMarkers(dc.corr,q_th = 0.001, lfc_th = 1.2,nDE = 10)
de$genes.df
```
