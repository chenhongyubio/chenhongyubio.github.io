---
layout:     post   				    # 使用的布局（不需要改）
title:      ExpressionSet对象			# 标题 
subtitle:   ExpressionSet对象理解学习 #副标题
date:       2020-01-11 				# 时间
author:     CHY					# 作者
header-img: img/wallhaven-eypv1k.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - 生信分析
---
## 基本介绍
ExpressionSet对象主要是对表达矩阵加上样本分组信息的封装，专门将多种不同来源的数据整合到一起方便数据处理的一种数据类型，简单的说，ExpressionSet 把表达数据（assayData 存储芯片、测序等表达数据），表型信息（phenoData 存储样本信息），注释等元信息（featureData, annotation 存储芯片或者测序技术的元数据），以及操作流程（protocolData 存储样本处理相关的信息，通常由厂家提供）和实验（experimentData 用来描述实验相关信息）几种密切相关数据封装在一起，由Biobase包引入。
重点就是 `exprs` 函数提取表达矩阵，`pData` 函数看看该对象的样本分组信息。

用GEOquery包来下载得到 ExpressionSet 对象
gse1009=GEOquery::getGEO("GSE1009")

## 从头构建
利用convert包的 as 转换
```
library("convert")
# as(object,"ExpressionSet")

# 准备Assay data(相当于表达矩阵)
# 行为基因，列为样本
minimalSet <- ExpressionSet(assayData = exprs)  

# 准备Phenotypic data
# 主要是描述样本的相关信息（表型数据）
# 行为样本，列为表型变量
# assayData 和 pData 之间存在对应关系
# AnnotatedDataFrame 数据类型把 pData 和 metadata 封装起来
phenoData <- new("AnnotatedDataFrame", data = pData, varMetadata = metadata)
phenoData

# 准备Annotations and feature data
通常芯片的注释文件都是单独的 Bioconductor 包。这些注释文件描述了探针与基因、基因的功能等等之间的对应关系，有时候还有有 GO 和 KEGG 等其他来源的信息。annotate 和 AnnotationDbi 包就是用来处理这些注释元数据包的。
annotation <- "hgu95av2"

# 准备Experiment description
# 主要提供一些研究者以及实验室等信息
experimentData <- new(
  "MIAME",
  name = "Pierre Fermat",
  lab = "Francis Galton Lab",
  contact = "pfermat@lab.not.exist",
  title = "Smoking-Cancer Experiment",
  abstract = "An example ExpressionSet",
  url = "www.lab.not.exist",
  other = list(notes = "Created from text files")
)

# 构建ExpressionSet对象
exampleSet <- ExpressionSet(
  assayData = exprs,
  phenoData = phenoData,
  experimentData = experimentData,
  annotation = annotation
)
exampleSet

# 常用查看函数
# $符号即可
featureNames()
sampleNames()
varLabels()
exprs()  #提取表达矩阵

# 取子集
vv <- exampleSet[1:5, 1:3]
males <- exampleSet[ , exampleSet$gender == "Male"]
```

## 参考链接
[Bioconductor 中的 ExpressionSet 数据类型](https://jiangjun.link/post/2019/03/bioconductor-expressionset/)