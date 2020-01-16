---
layout:     post   				    # 使用的布局（不需要改）
title:      RNA-velocity(2)				# 标题 
subtitle:   进行单细胞RNA-velocity分析的其他方法 #副标题
date:       2020-01-16 				# 时间
author:     CHY					# 作者
header-img: img/wallhaven-2026.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - 单细胞
---

## Seurat直接用于RNA velocity分析
```
# 安装对应的R包
library(Seurat)
library(velocyto.R)
library(SeuratWrappers)

# 准备数据构建RNA velocity
# If you don't have velocyto's example mouse bone marrow dataset, download with the CURL command
# curl::curl_download(url = 'http://pklab.med.harvard.edu/velocyto/mouseBM/SCG71.loom', destfile
# = '~/Downloads/SCG71.loom')
ldat <- ReadVelocity(file = "~/Downloads/SCG71.loom")
bm <- as.Seurat(x = ldat)
bm <- SCTransform(object = bm, assay = "spliced")
bm <- RunPCA(object = bm, verbose = FALSE)
bm <- FindNeighbors(object = bm, dims = 1:20)
bm <- FindClusters(object = bm)
bm <- RunUMAP(object = bm, dims = 1:20)
bm <- RunVelocity(object = bm, deltaT = 1, kCells = 25, fit.quantile = 0.02)
ident.colors <- (scales::hue_pal())(n = length(x = levels(x = bm)))
names(x = ident.colors) <- levels(x = bm)
cell.colors <- ident.colors[Idents(object = bm)]
names(x = cell.colors) <- colnames(x = bm)
show.velocity.on.embedding.cor(emb = Embeddings(object = bm, reduction = "umap"), vel = Tool(object = bm, 
    slot = "RunVelocity"), n = 200, scale = "sqrt", cell.colors = ac(x = cell.colors, alpha = 0.5), 
    cex = 0.8, arrow.scale = 3, show.grid.flow = TRUE, min.grid.cell.mass = 0.5, grid.n = 40, arrow.lwd = 1, 
    do.par = FALSE, cell.border.alpha = 0.1)
```

## scVelo estimates RNA velocities
剪接和未剪接矩阵需要通过其他方式获得  
[velocyto](http://velocyto.org/velocyto.py/tutorial/cli.html)  
[kallisto pipeline via loompy](https://linnarssonlab.org/loompy/kallisto/index.html)  
[RNA velocity tutorial with kb](https://www.kallistobus.tools/kb_velocity_tutorial.html)  
```
# 需要在Python3.6以上运行
import scvelo as scv
scv.settings.set_figure_params('scvelo')   # 设置matplotlib settings可视化
adata = scv.read(filename, cache=True)  # 读入数据(loom/h5ad/csv格式)
```
![数据结构](https://github.com/chenhongyubio/chenhongyubio.github.io/blog/master/img/2026-1.png)
```
# 如果之前已经存在adata对象，可以直接merge剪接未剪接矩阵
ldata = scv.read(filename.loom, cache=True)
adata = scv.utils.merge(adata, ldata)
# 也可以试试内置数据集
adata = scv.datasets.dentategyrus()

# 预处理preprocessing (scv.pp.*)
scv.pp.filter_and_normalize(adata, **params)
scv.pp.moments(adata, **params)

# Velocity分析(scv.tl.*)
scv.tl.velocity(adata, mode='stochastic', **params)
# 不同模型针对不同参数,deterministic model is obtained by setting mode='deterministic',the dynamical model is obtained by setting mode='dynamical',
# 数据存储在adata.layers中。
scv.tl.velocity_graph(adata, **params)

# 可视化绘图
scv.pl.velocity_embedding(adata, basis='umap', **params)
scv.pl.velocity_embedding_grid(adata, basis='umap', **params)
scv.pl.velocity_embedding_stream(adata, basis='umap', **params)
scv.pl.velocity(adata, var_names=['gene_A', 'gene_B'], **params)
scv.pl.velocity_graph(adata, **params)
```


## 参考链接
[Estimating RNA Velocity using Seurat](http://htmlpreview.github.io/?https://github.com/satijalab/seurat-wrappers/blob/master/docs/velocity.html)