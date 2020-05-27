---
layout:     post   				    # 使用的布局（不需要改）
title:      SingleCellExperiment对象				# 标题 
subtitle:   单细胞数据存储对象 #副标题
date:       2020-05-27				# 时间
author:     CHY					# 作者
header-img: img/wallhaven-2051.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - 单细胞
---

对单细胞数据常用的数据结构SingleCellExperiment(简称SCE)进行学习，了解数据存储结构等。
主要参考：[Bioconductor中的教程](https://bioconductor.org/packages/devel/bioc/vignettes/SingleCellExperiment/inst/doc/intro.html) <br>

SCE对象特点：行为基因，列为细胞，同时还存储降维信息等。<br>
![对象数据结构](img/SingleCellExperiment.webp) <br>
不同的创建方式
```
# 创建SCE对象
library(SingleCellExperiment)
counts <- matrix(rpois(100, lambda = 10), ncol=10, nrow=10) # 输入文件为原始表达矩阵
sce <- SingleCellExperiment(counts)
sce
# 专门针对metadata进行设定
pretend.cell.labels <- sample(letters, ncol(counts), replace=TRUE)
pretend.gene.lengths <- sample(10000, nrow(counts))
sce <- SingleCellExperiment(list(counts=counts),
    colData=DataFrame(label=pretend.cell.labels),
    rowData=DataFrame(length=pretend.gene.lengths),
    metadata=list(study="GSE111111")
)
sce
# SCE对象是基于SummarizedExperiment对象而来
se <- SummarizedExperiment(list(counts=counts))
as(se, "SingleCellExperiment")
```

对象中数据提取
```
# 简单的提取
colData(sce)
rowData(sce)
counts(sce)
# 如果想提取出size facotr
colData(sce,internal = T)
# 如果想提取出spike-in
rowData(sce,internal = T)
```

SCE对象的核心在于assays,这是一个列表，其中包含了许多表达数据，例如原始数据count或者其他标准化处理过的数据。

