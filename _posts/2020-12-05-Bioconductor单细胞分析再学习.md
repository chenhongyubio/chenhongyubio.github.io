---
layout: post # 使用的布局（不需要改）
title:  Bioconductor单细胞分析再学习  # 标题
subtitle: Orchestrating Single-Cell Analysis with Bioconductor #副标题
date: 2020-12-05 # 时间
author: CHY # 作者
header-img: img/wallhaven-dpxqvo.jpg #这篇文章标题背景图片
catalog: true # 是否归档
tags: #标签
  - 单细胞
---

关于单细胞分析的内容，还存在许多的细节问题还没有思考明白，比如说细胞注释到底应该如何解决？或者在鉴定差异基因时候如何选择Marker gene？故在此重新学习单细胞分析教程，加深分析内容的理解。<br>

#### Orchestrating Single-Cell Analysis with Bioconductor
Orchestrating Single-Cell Analysis with Bioconductor教程主要分为三部分的内容：第一部分关注于学习R以及bioconductor；第二部分关注于单细胞分析中各个部分的内容；第三部分关注于整体的分析流程梳理。<br>

#### Data Infrastructure数据框架
SingleCellExperiment对象数据结构：
1. 构建基础的SingleCellExperiment对象结构，首先需要用表达矩阵填充**assays slot**；
```
# 表达矩阵在R中必须为matrix
sce <- SingleCellExperiment(assays = list(counts = counts_matrix))  
# 通过list添加，list每个元素代表一个矩阵

# 构建SingleCellExperiment对象后获取表达矩阵
assay(sce, "counts")
counts(sce)

# scater标准化数据后存储在SingleCellExperiment对象中
sce <- scater::logNormCounts(sce)
logcounts(sce)

# 查看SingleCellExperiment对象存放的数据
assays(sce)

# 自行添加数据到SingleCellExperiment对象中
counts_100 <- counts(sce) + 100
assay(sce, "counts_100") <- counts_100 # assign a new entry to assays slot
assays(sce)
```
2. 细胞Metadata存放在colData slot，输入为data.frame
```
# 添加cell metadata到colData slot
sce <- SingleCellExperiment(assays = list(counts = counts_matrix), colData = cell_metadata)
colData(sce)
sce$batch

# scater函数自动添加colData
sce <- scater::addPerCellQC(sce)
colData(sce)

# 自行添加colData
sce$more_stuff <- runif(ncol(sce))
colnames(colData(sce))

# colData常见用法--用做筛选
sce[, sce$batch == 1]
```

3. 基因Metadata存放在rowData中，输入为data.frame
```
# 获取及添加rowMetadata
rowRanges(sce)
sce <- scater::addPerFeatureQC(sce)
rowData(sce)

# 关于基因的注释信息可以通过AnnotationHub获取
library(AnnotationHub)
edb <- AnnotationHub()[["AH73881"]] # Human, Ensembl v97.Ensembl标识符
genes(edb)[,2]

# 提取特定基因
sce[c("gene_1", "gene_4"), ]
sce[c(1, 4), ]
```

4. 其余元数据存放在metadata slot，输入为list
```
my_genes <- c("gene_1", "gene_5")
metadata(sce) <- list(favorite_genes = my_genes)
metadata(sce)

# 增加metadata
your_genes <- c("gene_4", "gene_8")
metadata(sce)$your_genes <- your_genes
metadata(sce)
```

5. 降维数据存放在reducedDims slot中，包含一系列数字矩阵构成的list；其中矩阵行为细胞，列为维度
```
ce <- scater::logNormCounts(sce)
sce <- scater::runPCA(sce)
reducedDim(sce, "PCA")

# 查看所有内容
reducedDims(sce)

# 自行添加降维信息
u <- uwot::umap(t(logcounts(sce)), n_neighbors = 2)
reducedDim(sce, "UMAP_uwot") <- u
reducedDims(sce) # Now stored in the object.
```

6. 类似于spike-in的数据可以先构建SummarizedExperiment对象，然后再存储在SingleCellExperiment对象中
```
spike_counts <- cbind(cell_1 = rpois(5, 10), 
    cell_2 = rpois(5, 10), 
    cell_3 = rpois(5, 30))
rownames(spike_counts) <- paste0("spike_", 1:5)
spike_se <- SummarizedExperiment(list(counts=spike_counts))
spike_se
altExp(sce, "spike") <- spike_se
altExps(sce)
```

7. 每个细胞对应的sizeFactors()也可存放
```
sce <- scran::computeSumFactors(sce)
sizeFactors(sce)
# 自行添加
sizeFactors(sce) <- scater::librarySizeFactors(sce)
sizeFactors(sce)
```

#### Analysis Overview
1. 实验设计--测序方法选择、测序细胞数
2. 获取表达矩阵--CellRanger(10X)、alevin(Pseudo-alignment)、scPipe(highly multiplexed protocols)、scruff(CEL-seq)
3. 数据预处理--去除低质量细胞、normalized、挑选高变异细胞、降维、聚类
```
library(scRNAseq)
sce <- MacoskoRetinaData()

# Quality control.
library(scater)
is.mito <- grepl("^MT-", rownames(sce))
qcstats <- perCellQCMetrics(sce, subsets=list(Mito=is.mito))
filtered <- quickPerCellQC(qcstats, percent_subsets="subsets_Mito_percent")
sce <- sce[, !filtered$discard]

# Normalization.
sce <- logNormCounts(sce)

# Feature selection.
library(scran)
dec <- modelGeneVar(sce)
hvg <- getTopHVGs(dec, prop=0.1)

# Dimensionality reduction.
set.seed(1234)
sce <- runPCA(sce, ncomponents=25, subset_row=hvg)
sce <- runUMAP(sce, dimred = 'PCA', external_neighbors=TRUE)

# Clustering.
g <- buildSNNGraph(sce, use.dimred = 'PCA')
colLabels(sce) <- factor(igraph::cluster_louvain(g)$membership)

# Visualization.
plotUMAP(sce, colour_by="label")
```

#### Quality Control
低质量细胞的特征在于：1. low total counts; 2. few expressed genes; 3. high mitochondrial; 4. high spike-in proportions<br>
可能影响后续的结果：1. 形成了自己独特的簇，使对结果的解释变得复杂。2. 在方差估计或主成分分析过程中扭曲了异质性的特征；3. 包含的基因似乎由于主动缩放以针对小文库大小进行标准化而被强烈“上调”。<br>
鉴定低质量细胞的指标：1. library size 2. 表达基因数 3. spike-in 基因所占比例 4. 线粒体基因所占比例<br>
```
# Retrieving the mitochondrial transcripts using genomic locations included in
# the row-level annotation for the SingleCellExperiment.
location <- rowRanges(sce.416b)
is.mito <- any(seqnames(location)=="MT")

# ALTERNATIVELY: using resources in AnnotationHub to retrieve chromosomal
# locations given the Ensembl IDs; this should yield the same result.
library(AnnotationHub)
ens.mm.v97 <- AnnotationHub()[["AH73905"]]
chr.loc <- mapIds(ens.mm.v97, keys=rownames(sce.416b),
    keytype="GENEID", column="SEQNAME")
is.mito.alt <- which(chr.loc=="MT")

library(scater)
df <- perCellQCMetrics(sce.416b, subsets=list(Mito=is.mito))
df

# 保存QC结果到SingleCellExperiment对象
sce.416b <- addPerCellQC(sce.416b, subsets=list(Mito=is.mito))
colnames(colData(sce.416b))
```

**如何鉴定低质量细胞？**<br>
1. 采用固定的阈值
```
qc.lib <- df$sum < 1e5
qc.nexprs <- df$detected < 5e3
qc.spike <- df$altexps_ERCC_percent > 10
qc.mito <- df$subsets_Mito_percent > 10
discard <- qc.lib | qc.nexprs | qc.spike | qc.mito

# Summarize the number of cells removed for each reason.
DataFrame(LibSize=sum(qc.lib), NExprs=sum(qc.nexprs),
    SpikeProp=sum(qc.spike), MitoProp=sum(qc.mito), Total=sum(discard))
```
2. 采用自适应的阈值
   2.1 鉴定outliers，如采用3个MADs
```
qc.lib2 <- isOutlier(df$sum, log=TRUE, type="lower")
qc.nexprs2 <- isOutlier(df$detected, log=TRUE, type="lower")
attr(qc.lib2, "thresholds")    # 查看阈值

qc.spike2 <- isOutlier(df$altexps_ERCC_percent, type="higher")
attr(qc.spike2, "thresholds")

reasons <- quickPerCellQC(df, 
    sub.fields=c("subsets_Mito_percent", "altexps_ERCC_percent"))
colSums(as.matrix(reasons))
```

3. Outlier鉴定的假设是：大部分细胞都是高质量的；QC指标是独立于细胞生物信息外的
4. 当存在多个批次时，如果每个批次单独分开，则可以使用针对单独batch使用isOutlier()；当多个批次整合后，需要添加batch=参数，综合多个batch的阈值来进行筛选
```
library(scRNAseq)
sce.grun <- GrunPancreasData()
sce.grun <- addPerCellQC(sce.grun)

# First attempt with batch-specific thresholds.
discard.ercc <- isOutlier(sce.grun$altexps_ERCC_percent,
    type="higher", batch=sce.grun$donor)
with.blocking <- plotColData(sce.grun, x="donor", y="altexps_ERCC_percent",
    colour_by=I(discard.ercc))

# Second attempt, sharing information across batches
# to avoid dramatically different thresholds for unusual batches.
discard.ercc2 <- isOutlier(sce.grun$altexps_ERCC_percent,
    type="higher", batch=sce.grun$donor,
    subset=sce.grun$donor %in% c("D17", "D2", "D7"))
without.blocking <- plotColData(sce.grun, x="donor", y="altexps_ERCC_percent",
    colour_by=I(discard.ercc2))

gridExtra::grid.arrange(with.blocking, without.blocking, ncol=2)
```

5. 高维空间鉴定(多个QC指标，存在一定风险)
```
stats <- cbind(log10(df$sum), log10(df$detected),
    df$subsets_Mito_percent, df$altexps_ERCC_percent)

library(robustbase)
outlying <- adjOutlyingness(stats, only.outlyingness = TRUE)
multi.outlier <- isOutlier(outlying, type = "higher")
summary(multi.outlier)
```

6. 指标相关性揭示数据质量
```
colData(sce.416b) <- cbind(colData(sce.416b), df)
sce.416b$block <- factor(sce.416b$block)
sce.416b$phenotype <- ifelse(grepl("induced", sce.416b$phenotype),
    "induced", "wild type")
sce.416b$discard <- reasons$discard

gridExtra::grid.arrange(
    plotColData(sce.416b, x="block", y="sum", colour_by="discard",
        other_fields="phenotype") + facet_wrap(~phenotype) + 
        scale_y_log10() + ggtitle("Total count"),
    plotColData(sce.416b, x="block", y="detected", colour_by="discard", 
        other_fields="phenotype") + facet_wrap(~phenotype) + 
        scale_y_log10() + ggtitle("Detected features"),
    plotColData(sce.416b, x="block", y="subsets_Mito_percent", 
        colour_by="discard", other_fields="phenotype") + 
        facet_wrap(~phenotype) + ggtitle("Mito percent"),
    plotColData(sce.416b, x="block", y="altexps_ERCC_percent", 
        colour_by="discard", other_fields="phenotype") + 
        facet_wrap(~phenotype) + ggtitle("ERCC percent"),
    ncol=1
)

sce.zeisel <- addPerCellQC(sce.zeisel, 
    subsets=list(Mt=rowData(sce.zeisel)$featureType=="mito"))

qc <- quickPerCellQC(colData(sce.zeisel), 
    sub.fields=c("altexps_ERCC_percent", "subsets_Mt_percent"))
sce.zeisel$discard <- qc$discard

plotColData(sce.zeisel, x="sum", y="subsets_Mt_percent", colour_by="discard")
plotColData(sce.zeisel, x="altexps_ERCC_percent", y="subsets_Mt_percent",
    colour_by="discard")
```

7. 低质量细胞过滤
```
filtered <- sce.416b[,!reasons$discard]

# 查看QC去除的细胞是否集中于特定的细胞类群
# Using the 'discard' vector for demonstration purposes, 
# as it has more cells for stable calculation of 'lost'.
lost <- calculateAverage(counts(sce.416b)[,!discard])
kept <- calculateAverage(counts(sce.416b)[,discard])

library(edgeR)
logged <- cpm(cbind(lost, kept), log=TRUE, prior.count=2)
logFC <- logged[,1] - logged[,2]
abundance <- rowMeans(logged)
plot(abundance, logFC, xlab="Average count", ylab="Log-FC (lost/kept)", pch=16)
points(abundance[is.mito], logFC[is.mito], col="dodgerblue", pch=16)
```

8. 仅标注低质量细胞，但在后续分析中不去除
```
marked <- sce.416b
marked$discard <- batch.reasons$discard
```

QC比较担心的问题在于将一些生物现象也磨灭掉，但如果不去除，会极大影响后续分析的结果；最终还是建议进行一些基础的过滤操作；<br>

#### Normalization
测序深度的系统误差在单细胞中较为常见，主要来源于cDNA捕获及PCR扩增带来的系统误差；Normalization旨在去除这些误差，便于**细胞之间的比较**；以便于异质性或差异基因来源于生物差异。<br>
**Normalization occurs regardless of the batch structure and only considers technical biases, while batch correction - as the name suggests - only occurs across batches and must consider both technical biases and biological differences.**<br>
技术误差通常通过统一的模式影响所有基因；而生物差异则各不相同；

1. 文库大小标准化
   基于两个细胞间不存在差异不平衡的现象，但在单细胞数据中不是很常见；在后续的聚类及差异分析中够用了。
2. Normalization by deconvolution
   与DESeq2-estimateSizeFactorsFromMatrix 和 edgeR-calcNormFactors函数类似，假设大部分基因都不是差异基因，单细胞数据采用Pool-based size factors进行Normalization
```
library(scran)
set.seed(100)
clust.zeisel <- quickCluster(sce.zeisel) 
table(clust.zeisel)
deconv.sf.zeisel <- calculateSumFactors(sce.zeisel, cluster=clust.zeisel)
summary(deconv.sf.zeisel)
```
1. Normalization by spike-ins
```
library(scRNAseq)
sce.richard <- RichardTCellData()
sce.richard <- sce.richard[,sce.richard$`single cell quality`=="OK"]
sce.richard
sce.richard <- computeSpikeFactors(sce.richard, "ERCC")
summary(sizeFactors(sce.richard))
```

4. size factors应用
```
set.seed(100)
clust.zeisel <- quickCluster(sce.zeisel) 
sce.zeisel <- computeSumFactors(sce.zeisel, cluster=clust.zeisel, min.mean=0.1)
sce.zeisel <- logNormCounts(sce.zeisel)
assayNames(sce.zeisel)
```

#### Feature selection
1. 计算每个基因的变异性
   1.  simply compute the variance of the log-normalized expression values
   2.  squared coefficient of variation of the normalized expression values prior to log-transformation
   3.  定量技术差异（常规变异可能是由技术差异加上无用的生物差异，可以通过分析spike-in来鉴定准确的技术差异，当不存在spike-in时，可以对noise进行一些分布假设，如UMI数据近似泊松分布）
   4.   Accounting for blocking factors--解决批次的问题，选取在每个批次中都高表达的基因
2. 高变异基因选择
   1. 选择具有高变异的top X 基因
   2. 通过p-values选择
   3. 保留趋势线以上所有基因 -- 对于鉴定rare cell较为友好
   4. 选择感兴趣的先验基因
```
# 基因的差异更多由丰度造成，少部分由潜在生物差异造成
# 变异选择
# modelling of the mean-variance relationship
library(scran)
dec.pbmc <- modelGeneVar(sce.pbmc)
# Visualizing the fit:
fit.pbmc <- metadata(dec.pbmc)
plot(fit.pbmc$mean, fit.pbmc$var, xlab="Mean of log-expression",
    ylab="Variance of log-expression")
curve(fit.pbmc$trend(x), col="dodgerblue", add=TRUE, lwd=2)
# Ordering by most interesting genes for inspection.
dec.pbmc[order(dec.pbmc$bio, decreasing=TRUE),] 

# 对于出现过拟合的现象，可以设置参数进行调整(比如一些高丰度的基因也是高变异基因时)
sce.seger <- sce.seger[,sce.seger$Donor=="HP1507101"]
dec.default <- modelGeneVar(sce.seger)
dec.noweight <- modelGeneVar(sce.seger, density.weights=FALSE)  # 关闭density weights
fit.default <- metadata(dec.default)
plot(fit.default$mean, fit.default$var, xlab="Mean of log-expression",
    ylab="Variance of log-expression") 
curve(fit.default$trend(x), col="dodgerblue", add=TRUE, lwd=2)
fit.noweight <- metadata(dec.noweight)
curve(fit.noweight$trend(x), col="red", add=TRUE, lwd=2)
legend("topleft", col=c("dodgerblue", "red"), legend=c("Default", "No weight"), lwd=2)
```
```
# CV变异系数选择
#  Large CV2 values that deviate strongly from the trend are likely to represent genes affected by biological structure.
# the deviation from the trend in terms of the ratio of its CV2 to the fitted value of trend at its abundance
dec.cv2.pbmc <- modelGeneCV2(sce.pbmc)
fit.cv2.pbmc <- metadata(dec.cv2.pbmc)
plot(fit.cv2.pbmc$mean, fit.cv2.pbmc$cv2, log="xy")
curve(fit.cv2.pbmc$trend(x), col="dodgerblue", add=TRUE, lwd=2)
dec.cv2.pbmc[order(dec.cv2.pbmc$ratio, decreasing=TRUE),]
```
```
# 利用spike-in鉴定技术差异
dec.spike.416b <- modelGeneVarWithSpikes(sce.416b, "ERCC")
dec.spike.416b[order(dec.spike.416b$bio, decreasing=TRUE),]
plot(dec.spike.416b$mean, dec.spike.416b$total, xlab="Mean of log-expression",
    ylab="Variance of log-expression")
fit.spike.416b <- metadata(dec.spike.416b)
points(fit.spike.416b$mean, fit.spike.416b$var, col="red", pch=16)
curve(fit.spike.416b$trend(x), col="dodgerblue", add=TRUE, lwd=2)

# 无spike-in鉴定技术差异
set.seed(0010101)
dec.pois.pbmc <- modelGeneVarByPoisson(sce.pbmc)
dec.pois.pbmc <- dec.pois.pbmc[order(dec.pois.pbmc$bio, decreasing=TRUE),]
head(dec.pois.pbmc)
plot(dec.pois.pbmc$mean, dec.pois.pbmc$total, pch=16, xlab="Mean of log-expression",
    ylab="Variance of log-expression")
curve(metadata(dec.pois.pbmc)$trend(x), col="dodgerblue", add=TRUE)
```

CV2更适合于对一些在rare cell type中低丰度的高变异基因high rank；同时更适合原始数据而不是log转换后的数据

#### Dimensionality reduction
Perform the PCA on the log-normalized expression values.<br>
1. PC数量的选择--Elbow plot,
2. 保留那些代表差异达到特定阈值的PC，如解释80%差异的PC,也可以计算生物差异所占的比例
3. Based on population structure -- 类群与PC对应
4. Using random matrix theory
5. Count-based dimensionality reduction
```
# Elbow plot识别PC -- TOP PC应该比其余PC解释的差异性大得多
percent.var <- attr(reducedDim(sce.zeisel), "percentVar")
chosen.elbow <- PCAtools::findElbowPoint(percent.var)
chosen.elbow
plot(percent.var, xlab="PC", ylab="Variance explained (%)")
abline(v=chosen.elbow, col="red")
```
```
# 选择生物差异比例
# 计算可代表这些比例的PC
library(scran)
set.seed(111001001)
denoised.pbmc <- denoisePCA(sce.pbmc, technical=dec.pbmc, subset.row=top.pbmc)
ncol(reducedDim(denoised.pbmc))
```
```
# Based on population structure
pcs <- reducedDim(sce.zeisel)
choices <- getClusteredPCs(pcs)
val <- metadata(choices)$chosen
plot(choices$n.pcs, choices$n.clusters,
    xlab="Number of PCs", ylab="Number of clusters")
abline(a=1, b=1, col="red")
abline(v=val, col="grey80", lty=2)
reducedDim(sce.zeisel, "PCA.clust") <- pcs[,1:val]
```
```
# Using random matrix theory
# Generating more PCs for demonstration purposes:
set.seed(10100101)
sce.zeisel2 <- runPCA(sce.zeisel, subset_row=top.hvgs, ncomponents=200)
mp.choice <- PCAtools::chooseMarchenkoPastur(
    .dim=c(length(top.hvgs), ncol(sce.zeisel2)),
    var.explained=attr(reducedDim(sce.zeisel2), "varExplained"),
    noise=median(dec.zeisel[top.hvgs,"tech"]))
mp.choice

# Parallel analysis
set.seed(100010)
horn <- PCAtools::parallelPCA(logcounts(sce.zeisel)[top.hvgs,],
    BSPARAM=BiocSingular::IrlbaParam(), niters=10)
horn$n

plot(horn$original$variance, type="b", log="y", pch=16)
permuted <- horn$permuted
for (i in seq_len(ncol(permuted))) {
    points(permuted[,i], col="grey80", pch=16)
    lines(permuted[,i], col="grey80", pch=16)
}
abline(v=horn$n, col="red")

gv.choice <- PCAtools::chooseGavishDonoho(
    .dim=c(length(top.hvgs), ncol(sce.zeisel2)),
    var.explained=attr(reducedDim(sce.zeisel2), "varExplained"),
    noise=median(dec.zeisel[top.hvgs,"tech"]))
gv.choice
```
```
# 


```