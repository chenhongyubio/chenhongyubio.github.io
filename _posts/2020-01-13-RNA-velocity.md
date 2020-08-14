---
layout: post # 使用的布局（不需要改）
title: RNA-velocity # 标题
subtitle: RNA-velocity相关软件流程分析 #副标题
date: 2020-01-13 # 时间
author: CHY # 作者
header-img: img/wallhaven-2023.jpg #这篇文章标题背景图片
catalog: true # 是否归档
tags: #标签
  - 单细胞
---

## RNA velocity 基本概念

RNA velocity：the time derivative of the gene expression state can be directly estimated by distinguishing between unspliced and spliced mRNAs in common single-cell RNA sequencing protocols. a high-dimensional vector that predicts the future state of individual cells on a timescale of hours.
也就是通过区分普通单细胞 RNA 测序中未剪接和剪接的 mrna 在小时的时间尺度上来直接估计预测单个细胞的未来状态。换句话说通过对 unspliced(nascent) 和 spliced(mature) mRNA 丰度的评估在时间维度上揭示转录本动态变化。

## velocyto

目前有 Python 和 R 两个版本可以运行。

#### velocyto.py

前提要求：Python >= 3.6.0
Velocyto 主要由两部分组成：
A command line interface(CLI)，用于生成拼接/非拼接表达式矩阵。
A library：包含从上述数据矩阵估计 RNA 速度的函数的库

```
## 安装依赖的库
conda install numpy scipy cython numba matplotlib scikit-learn h5py click
pip install pysam

## 安装velocyto
pip install velocyto
velocyto --help  ## 验证是否安装成功
## 遇到了报错，
pip install --upgrade cython

## 运行CLI
## 输入bam/sam文件运行生成loom文件(包含spliced/unspliced/ambiguous的count)
## 第一步：准备基因组注释文件 (从GENCONE、Ensembl或者对应物种数据库下载gtf文件)
## 第二步：准备表达重复注释(这一步不是必须的操作，可能也不太好找，试试UCSC等数据库)

## 第三步：运行velocyto获得Loom文件
## 主要命令为velocyto run，针对不同的数据可以使用run10x, run_smartseq2, run_dropest
velocyto run10x -m repeat_msk.gtf mypath/sample01 somepath/refdata-cellranger-mm10-1.2.0/genes/genes.gtf  ## 针对10x样本
velocyto run_smartseq2 -o OUTPUT -m repeat_msk.gtf -e MyTissue plateX/*/*.bam mm10_annotation.gtf  ## 针对smartseq2多个样本分析整合结果到单个loom文件(bam文件需要按照位置排序)
velocyto run_dropest -o ~/mydata/SRR5945695_results -m rep_mask.gtf ~/mydata/SRR5945695_1/correct_SRR5945695_1Aligned.sortedByCoord.out.tagged.bam mm10_annotation.gtf  ## 针对Dropseq数据(前期数据准备过程需要按照对应DropEst来进行)
velocyto run -b filtered_barcodes.tsv -o output_path -m repeat_msk_srt.gtf possorted_genome_bam.bam mm10_annotation.gtf ## 针对任何技术
## 生成的loom文件为HDF5文件，可以使用loompy包进行处理
loompy.combine(files, output_filename, key="Accession")

## 分析Loom文件
import velocyto as vcy
vlm = vcy.VelocytoLoom("YourData.loom")
## 初步过滤(去除掉一些低未剪切RNA检出的细胞)
vlm.filter_cells(bool_array=vlm.initial_Ucell_size > np.percentile(vlm.initial_Ucell_size, 0.5))
## 选择合适的feature进行后续分析
vlm.set_clusters(vlm.ca["ClusterName"])
vlm.score_detection_levels(min_expr_counts=40, min_cells_express=30)
vlm.filter_genes(by_detection_levels=True)
vlm.score_cv_vs_mean(3000, plot=True, max_expr_avg=35) # feature选择
vlm.filter_genes(by_cv_vs_mean=True)
## 通过size进行标准化
vlm._normalize_S(relative_size=vlm.S.sum(0),
             target_size=vlm.S.sum(0).mean())
vlm._normalize_U(relative_size=vlm.U.sum(0),
             target_size=vlm.U.sum(0).mean())

## 拟合准备
vlm.perform_PCA()
vlm.knn_imputation(n_pca_dims=20, k=500, balanced=True, b_sight=3000, b_maxl=1500, n_jobs=16)

## 拟合分析
vlm.fit_gammas()
vlm.plot_phase_portraits(["Igfbpl1", "Pdgfra"]) # 绘图查看
vlm.predict_U() # 计算推断细胞的命运
vlm.calculate_velocity()
vlm.calculate_shift(assumption="constant_velocity")
vlm.extrapolate_cell_at_t(delta_t=1.)

## 将velocity投影到低维(embeddings),进行可视化
## tsne运行相当耗时
from sklearn.manifold import TSNE
bh_tsne = TSNE()
vlm.ts = bh_tsne.fit_transform(vlm.pcs[:, :25])
vlm.estimate_transition_prob(hidim="Sx_sz", embed="ts", transform="sqrt", psc=1,
                             n_neighbors=3500, knn_random=True, sampled_fraction=0.5)
vlm.calculate_embedding_shift(sigma_corr = 0.05, expression_scaling=True)
vlm.calculate_grid_arrows(smooth=0.8, steps=(40, 40), n_neighbors=300)
plt.figure(None,(20,10))
vlm.plot_grid_arrows(quiver_scale=0.6,
                    scatter_kwargs_dict={"alpha":0.35, "lw":0.35, "edgecolor":"0.4", "s":38, "rasterized":True}, min_mass=24, angles='xy', scale_units='xy',
                    headaxislength=2.75, headlength=5, headwidth=4.8, minlength=1.5,
                    plot_random=True, scale_type="absolute")
```

#### velocyto.R

这里借助于[RNA velocity with kallisto | bus and velocyto.R](https://bustools.github.io/BUS_notebooks_R/velocity.html)来学习 velocyto.R，因为官方的对我来说实在不太好理解。
kallisto | bus pipeline 可以更快的获得剪接的或者未剪接的转录本信息。
[kallisto](http://pachterlab.github.io/kallisto/manual.html)是一款快速比对软件，其最大的特点是不需要完整的参考基因组，但需要对物种的全转录本序列建立索引，再进行假比对。
软件要求：[kallisto](https://pachterlab.github.io/kallisto/download) >= 0.46；[bustools](https://github.com/BUStools/bustools/releases) >= 0.39.3 bustools 主要是针对 10X 数据分析，旨在替代 cellranger。
R 包：BUSpaRse:转转录本为基因文件给 bustools,同时读入 bustools 结果到 R
Seurat: [SeuratWrappers](https://github.com/satijalab/seurat-wrappers)直接分析 RNA velocity
velocyto.R：计算并可视化 RNA velocity

##### kallisto | bustools

```
# 运行命令
# 从基因组和基因组注释开始，建立转录组索引
kb ref -i transcriptome.idx -g transcripts_to_genes.txt -f1 cdna.fa dna.primary_assembly.fa.gz gtf.gz

# kb count uses kallisto to pseudoalign reads and bustools to quantify the data
kb count -i index.idx -g t2g.txt -x 10xv2 --h5ad -t 2 read_1.fastq.gz read_2.fastq.gz

```

```
## 安装
library(devtools)
install_github("velocyto-team/velocyto.R")
# Install devtools if it's not already installed
if (!require(devtools)) {
  install.packages("devtools")
}
# Install from GitHub
devtools::install_github("BUStools/BUSpaRse")
devtools::install_github("satijalab/seurat-wrappers")
devtools::install_github("velocyto-team/velocyto.R")
if (!require(BiocManager)) {
  install.packages("BiocManager")
}
BiocManager::install(c("DropletUtils", "BSgenome.Mmusculus.UCSC.mm10", "AnnotationHub"))
library(BUSpaRse)
library(Seurat)
library(SeuratWrappers)
library(BSgenome.Mmusculus.UCSC.mm10)
library(AnnotationHub)
library(zeallot) # For %<-% that unpacks lists in the Python manner
library(DropletUtils)
library(tidyverse)
library(uwot) # For umap
library(GGally) # For ggpairs
library(velocyto.R)
library(SingleR)
library(scales)
library(plotly)
theme_set(theme_bw())

# Download data
if (!file.exists("./data/neuron_10k_v3_fastqs.tar")) {
  download.file("http://s3-us-west-2.amazonaws.com/10x.files/samples/cell-exp/3.0.0/neuron_10k_v3/neuron_10k_v3_fastqs.tar", "./data/neuron_10k_v3_fastqs.tar", method = "wget", quiet = TRUE)
}
cd ./data
tar -xvf ./neuron_10k_v3_fastqs.tar

# 生成剪切矩阵
# 下载注释文件，也可以使用biomartr包的getGTF和getGFF文件
ah <- AnnotationHub()
query(ah, pattern = c("Ensembl", "97", "Mus musculus", "EnsDb"))
edb <- ah[["AH73905"]]
get_velocity_files(edb, L = 91, Genome = BSgenome.Mmusculus.UCSC.mm10,
                   out_path = "./output/neuron10k_velocity",
                   isoform_action = "separate")
# X:基因组注释文件edb；L:reads长度，10x V1,V2是98nt;V3是91nt.
# Genome参数：DNAStringSet或者BSgenome对象，可以从对应基因组R包获取，也可以从Ensembl, RefSeq, or GenBank with biomartr::getGenome
# Transcriptome：可以从基因组中提取
# isoform_action：两种选择，基因亚型来自于选择性剪接或转录起始或终止位点

# Intron index构建index（至少需要50G内存）
kallisto index -i ./output/mm_cDNA_introns_97.idx ./output/neuron10k_velocity/cDNA_introns.fa
cd ./data/neuron_10k_v3_fastqs
kallisto bus -i ../../output/mm_cDNA_introns_97.idx \
-o ../../output/neuron10k_velocity -x 10xv3 -t8 \
neuron_10k_v3_S1_L002_R1_001.fastq.gz neuron_10k_v3_S1_L002_R2_001.fastq.gz \
neuron_10k_v3_S1_L001_R1_001.fastq.gz neuron_10k_v3_S1_L001_R2_001.fastq.gz
cp ~/cellranger-3.0.2/cellranger-cs/3.0.2/lib/python/cellranger/barcodes/3M-february-2018.txt.gz \
./data/whitelist_v3.txt.gz
# Decompress
gunzip ./data/whitelist_v3.txt.gz
# 去掉那些少reads的barcodes
cp ~/cellranger-3.0.2/cellranger-cs/3.0.2/lib/python/cellranger/barcodes/3M-february-2018.txt.gz \
./data/whitelist_v3.txt.gz
# Decompress
gunzip ./data/whitelist_v3.txt.gz
## checks the whitelist and can correct some barcodes not on the whitelist but might have been due to sequencing error or mutation.(也可不运行)
cd ./output/neuron10k_velocity
bustools correct -w ../../data/whitelist_v3.txt -p output.bus | \
bustools sort -o output.correct.sort.bus -t4 -
bustools capture -s -x -o spliced.bus -c ./introns_tx_to_capture.txt -e matrix.ec -t transcripts.txt output.correct.sort.bus
bustools capture -s -x -o unspliced.bus -c ./cDNA_tx_to_capture.txt -e matrix.ec -t transcripts.txt output.correct.sort.bus
# 产生两个矩阵(剪接的和未剪接的)
cd ./output/neuron10k_velocity
bustools count -o unspliced -g ./tr2g.tsv -e matrix.ec -t transcripts.txt --genecounts unspliced.bus
bustools count -o spliced -g ./tr2g.tsv -e matrix.ec -t transcripts.txt --genecounts spliced.bus

# 预处理preprocessing
# Remove empty droplets去除空的油滴
c(spliced, unspliced) %<-% read_velocity_output(spliced_dir = "./output/neuron10k_velocity",
                                                spliced_name = "spliced",
                                                unspliced_dir = "./output/neuron10k_velocity",
                                                unspliced_name = "unspliced")
sum(unspliced@x) / (sum(unspliced@x) + sum(spliced@x))
tot_count <- Matrix::colSums(spliced)
summary(tot_count)
# 判断空油滴
bc_rank <- barcodeRanks(spliced)
bc_uns <- barcodeRanks(unspliced)
# 绘制knee plot图
tibble(rank = bc_rank$rank, total = bc_rank$total, matrix = "spliced") %>%
  bind_rows(tibble(rank = bc_uns$rank, total = bc_uns$total, matrix = "unspliced")) %>%
  distinct() %>%
  ggplot(aes(total, rank, color = matrix)) +
  geom_line() +
  geom_vline(xintercept = metadata(bc_rank)$knee, color = "blue", linetype = 2) +
  geom_vline(xintercept = metadata(bc_rank)$inflection, color = "green", linetype = 2) +
  geom_vline(xintercept = metadata(bc_uns)$knee, color = "purple", linetype = 3) +
  geom_vline(xintercept = metadata(bc_uns)$inflection, color = "cyan", linetype = 3) +
  annotate("text", y = c(1000, 1000, 500, 500),
           x = 1.5 * c(metadata(bc_rank)$knee, metadata(bc_rank)$inflection,
                       metadata(bc_uns)$knee, metadata(bc_uns)$inflectio),
           label = c("knee (s)", "inflection (s)", "knee (u)", "inflection (u)"),
           color = c("blue", "green", "purple", "cyan")) +
  scale_x_log10() +
  scale_y_log10() +
  labs(y = "Rank", x = "Total UMI counts") +
  theme_bw()
# 为了更好的选择，也可以绘制3D图
# Can only plot barcodes with both spliced and unspliced counts
bcs_inter <- intersect(colnames(spliced), colnames(unspliced))
s <- colSums(spliced[,bcs_inter])
u <- colSums(unspliced[,bcs_inter])
# Grid points
sr <- sort(unique(exp(round(log(s)*100)/100)))
ur <- sort(unique(exp(round(log(u)*100)/100)))
# Run naive approach
bc2 <- bc_ranks2(s, u, sr, ur)
# can't turn color to lot scale unless log values are plotted
z_use <- log10(bc2)
z_use[is.infinite(z_use)] <- NA
plot_ly(x = sr, y = ur, z = z_use) %>% add_surface() %>%
  layout(scene = list(xaxis = list(title = "Total spliced UMIs", type = "log"),
                      yaxis = list(title = "Total unspliced UMIs", type = "log"),
                      zaxis = list(title = "Rank (log10)")))
bcs_use <- colnames(spliced)[tot_count > metadata(bc_rank)$inflection]
# Remove genes that aren't detected
tot_genes <- Matrix::rowSums(spliced)
genes_use <- rownames(spliced)[tot_genes > 0]
sf <- spliced[genes_use, bcs_use]
uf <- unspliced[genes_use, bcs_use]
dim(sf)

## 细胞类型注释
# 利用SingleR使用分离的已知细胞类型的RNA-seq数据作为注释细胞类型的参考
# Get gene names
gns <- tr2g_EnsDb(edb)[,c("gene", "gene_name")] %>%
  distinct()
data("mouse.rnaseq")
# Convert from gene symbols to Ensembl gene ID
ref_use <- mouse.rnaseq$data
rownames(ref_use) <- gns$gene[match(rownames(ref_use), gns$gene_name)]
ref_use <- ref_use[!is.na(rownames(ref_use)),]
annot <- SingleR("single", sf, ref_data = ref_use, types = mouse.rnaseq$types)
ind <- annot$labels %in% c("NPCs", "Neurons", "OPCs", "Oligodendrocytes",
                           "qNSCs", "aNSCs", "Astrocytes", "Ependymal")
cells_use <- annot$cell.names[ind]
sf <- sf[, cells_use]
uf <- uf[, cells_use]

## 质量控制
# 剪接和未剪接矩阵都是需要归一化和缩放SCTransform(一个命令代替NormalizeData，ScaleData和FindVariableFeatures)
seu <- CreateSeuratObject(sf, assay = "sf") %>%
  SCTransform(assay = "sf", new.assay.name = "spliced")
seu[["uf"]] <- CreateAssayObject(uf)
seu <- SCTransform(seu, assay = "uf", new.assay.name = "unspliced")
# Add cell type metadata
seu <- AddMetaData(seu, setNames(annot$labels[ind], cells_use),
                   col.name = "cell_type")
cols_use <- c("nCount_sf", "nFeature_sf", "nCount_uf", "nFeature_uf")
VlnPlot(seu, cols_use, pt.size = 0.1, ncol = 1, group.by = "cell_type")
# Helper functions for ggpairs
log10_diagonal <- function(data, mapping, ...) {
  ggally_densityDiag(data, mapping, ...) + scale_x_log10()
}
log10_points <- function(data, mapping, ...) {
  ggally_points(data, mapping, ...) + scale_x_log10() + scale_y_log10()
}
ggpairs(seu@meta.data, columns = cols_use,
        upper = list(continuous = "cor"),
        diag = list(continuous = log10_diagonal),
        lower = list(continuous = wrap(log10_points, alpha = 0.1, size=0.3)),
        progress = FALSE)

## 降维PCA
DefaultAssay(seu) <- "spliced"
seu <- RunPCA(seu, verbose = FALSE, npcs = 70)
ElbowPlot(seu, ndims = 70)
DimPlot(seu, reduction = "pca",
        group.by = "cell_type", pt.size = 0.5, label = TRUE, repel = TRUE) +
  scale_color_brewer(type = "qual", palette = "Set2")
## 或者运行tsne
seu <- RunTSNE(seu, dims = 1:50, verbose = FALSE)
DimPlot(seu, reduction = "tsne",
        group.by = "cell_type", pt.size = 0.5, label = TRUE, repel = TRUE) +
  scale_color_brewer(type = "qual", palette = "Set2")
## 运行umap
seu <- RunUMAP(seu, dims = 1:50, umap.method = "uwot")
DimPlot(seu, reduction = "umap",
        group.by = "cell_type", pt.size = 0.5, label = TRUE, repel = TRUE) +
  scale_color_brewer(type = "qual", palette = "Set2")
seu <- FindNeighbors(seu, verbose = FALSE) %>%
  FindClusters(resolution = 1, verbose = FALSE) # Louvain
DimPlot(seu, pt.size = 0.5, reduction = "umap", label = TRUE)


## RNA velocity分析
seu <- RunVelocity(seu, ncores = 10, reduction = "pca", verbose = FALSE)
cell_pal <- function(cell_cats, pal_fun) {
  categories <- sort(unique(cell_cats))
  pal <- setNames(pal_fun(length(categories)), categories)
  pal[cell_cats]
}
label_clusters <- function(labels, coords, ...) {
  df <- tibble(label = labels, x = coords[,1], y = coords[,2])
  df <- df %>%
    group_by(label) %>%
    summarize(x = median(x), y = median(y))
  text(df$x, df$y, df$label, ...)
}
### 设置颜色矢量
cell_colors <- cell_pal(seu$cell_type, brewer_pal("qual", "Set2"))
cell_colors_clust <- cell_pal(seu$seurat_clusters, hue_pal())
names(cell_colors) <- names(cell_colors_clust) <- colnames(sf)
### 映射箭头
cc_umap <- show.velocity.on.embedding.cor(emb = Embeddings(seu, "umap"),
                                          vel = Tool(seu, slot = "RunVelocity"),
                                          n.cores = 50, show.grid.flow = TRUE,
                                          grid.n = 50, cell.colors = cell_colors,
                                          cex = 0.5, cell.border.alpha = 0,
                                          arrow.scale = 2, arrow.lwd = 0.6,
                                          xlab = "UMAP1", ylab = "UMAP2")
label_clusters(seu$cell_type, Embeddings(seu, "umap"), font = 2, col = "brown")
show.velocity.on.embedding.cor(emb = Embeddings(seu, "umap"),
                               vel = Tool(seu, slot = "RunVelocity"),
                               n.cores = 50, show.grid.flow = TRUE,
                               grid.n = 50, cell.colors = cell_colors_clust,
                               cex = 0.5, cell.border.alpha = 0,
                               arrow.scale = 2, arrow.lwd = 0.6,
                               cc = cc_umap$cc,
                               xlab = "UMAP1", ylab = "UMAP2")
label_clusters(seu$seurat_clusters, Embeddings(seu, "umap"), font = 2, cex = 1.2)


# 绘制特定基因热图
gene.relative.velocity.estimates(GetAssayData(seu, slot = "data", assay = "spliced"),
                                 GetAssayData(seu, slot = "data", assay = "unspliced"),
                                 cell.emb = Embeddings(seu, "umap"),
                                 show.gene = gns$gene[gns$gene_name == "Mef2c"],
                                 old.fit = Tool(seu, slot = "RunVelocity"),
                                 cell.colors = cell_colors)
```

## 参考链接

[RNA velocity of single cell](https://www.nature.com/articles/s41586-018-0414-6)

[velocyto](http://velocyto.org/)

[velocyto.py](http://velocyto.org/velocyto.py/index.html)
