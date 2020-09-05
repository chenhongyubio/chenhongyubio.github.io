---
layout: post # 使用的布局（不需要改）
title: GO_KEGG信息获取 # 标题
subtitle: 基因ID及GO KEGG信息获取 #副标题
date: 2020-09-04 # 时间
author: CHY # 作者
header-img: img/wallhaven-mdjw9k.png #这篇文章标题背景图片
catalog: true # 是否归档
tags: #标签
  - 生信分析
---

#### 多种方式获取基因对应所在 GO term 信息

获取所有基因对应的 GO 注释信息。以水稻为例

```
# 从NCBI Gene 数据库进行下载
# ftp://ftp.ncbi.nih.gov/gene/DATA/
# gene2go就是基因对应的GO注释文件，这个文件包含了所有物种的GO信息，可以根据物种对应的tax id提取指定物种。
# NCBI中用Entrez Id 标识每个基因，通过另外的几个文件，可以得到Entrez ID, Ensemble Id, Gene Symbol对应的GO信息。
```

```
# 从Bioconductor 获取
k <- keys(org.Hs.eg.db, keytype = "ENTREZID")[1:3]
select(org.Hs.eg.db,
   keys = k,
   columns = c("GO", "ONTOLOGY"),
   keytype="ENTREZID")

```

#### 获取基因对应所在 KEGG term 信息

```
library(biomaRt)
library(clusterProfiler)
# 根据clusterProfiler获取KEGG与基因对应关系

## download_KEGG下载物种对应KEGG数据库
osa_kegg <- clusterProfiler::download_KEGG("osa")
names(osa_kegg)
# 存在两个列表，一个是KEGG的通路编号和基因编号的关系，另一个是KEGG通路编号和名字的关系

# 合并数据表格
PATH2ID <- osa_kegg$KEGGPATHID2EXTID
PATH2NAME <- osa_kegg$KEGGPATHID2NAME
PATH_ID_NAME <- merge(PATH2ID, PATH2NAME, by="from")
colnames(PATH_ID_NAME) <- c("KEGGID", "ENTREZID", "DESCRPTION")

# ENTREZID转换--getBM函数
#在ensemble plants上能看到所有已提交的物种信息
ensembl = useMart(biomart = "plants_mart",host = "http://plants.ensembl.org")
#查看ensemble plants都有哪些物种信息，并设置为该物种信息。
dataset <- listDatasets(mart = ensembl)
head(dataset)
ensembl = useMart(biomart = "plants_mart",host = "http://plants.ensembl.org",dataset="osativa_eg_gene")
#查看该dataset上都有哪些属性，方便后面做添加
attributes <- listAttributes(ensembl)

# 转换成Ensemble ID
supplement <- getBM(attributes =c("ensembl_gene_id",'external_gene_name',"description"),filters = "ensembl_gene_id",values = a,mart = ensembl)
# 转换成GO ID并附上GO描述
supplements <- getBM(attributes =c("ensembl_gene_id",'go_id','goslim_goa_description'),
 filters = "ensembl_gene_id",values = a,mart = ensembl)
# 转换成NCBI ID
supplements <- getBM(attributes =c("ensembl_gene_id",'entrezgene_id'),
 filters = "ensembl_gene_id",values = a,mart = ensembl)

write.table(PATH_ID_NAME, "HSA_KEGG.txt", sep="\t")
```
