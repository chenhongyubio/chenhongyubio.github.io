---
layout: post # 使用的布局（不需要改）
title: RcisTarget # 标题
subtitle: RcisTarget #副标题
date: 2020-09-06 # 时间
author: CHY # 作者
header-img: img/wallhaven-p86vrj.jpg #这篇文章标题背景图片
catalog: true # 是否归档
tags: #标签
  - 生信分析
---

RcisTarget is an R-package to identify transcription factor (TF) binding motifs over-represented on a gene list.<br>
基于转录因子靶基因的 TSS 位点上下游，是否具有该转录因子结合的 motif 富集，将靶基因区分为直接靶基因和间接靶基因，从而保留直接靶基因。<br>

#### 安装

```
if (!requireNamespace("BiocManager", quietly=TRUE))
    install.packages("BiocManager")
# To support paralell execution:
BiocManager::install(c("doMC", "doRNG"))
# For the examples in the follow-up section of the tutorial:
BiocManager::install(c("DT", "visNetwork"))
BiocManager::install(RcisTarget)
```

#### workflow

输入：Gene list and motif databases( which should be chosen depending on the organism and the search space around the TSS of the genes) <br>

```
library(RcisTarget)

# Load gene sets to analyze. e.g.:
geneList1 <- read.table(file.path(system.file('examples', package='RcisTarget'), "hypoxiaGeneSet.txt"), stringsAsFactors=FALSE)[,1]
geneLists <- list(geneListName=geneList1)    # 构建为列表

# Select motif database to use (i.e. organism and distance around TSS)
data(motifAnnotations_hgnc)
motifRankings <- importRankings("~/databases/hg19-tss-centered-10kb-7species.mc9nr.feather")

# Motif enrichment analysis:
motifEnrichmentTable_wGenes <- cisTarget(geneLists, motifRankings,
                               motifAnnot=motifAnnotations_hgnc)
```

#### 需要的数据库

1. Gene-motif rankings: which provides the rankings (~score) of all the genes for each motif.
2. The annotation of motifs to transcription factors.
