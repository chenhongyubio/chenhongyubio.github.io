---
layout: post # 使用的布局（不需要改）
title: Mutational Signatures分析 # 标题
subtitle: 吸烟机电子烟测试潜在分析方向 #副标题
date: 2020-07-28 # 时间
author: CHY # 作者
header-img: img/wallhaven-MS.jpg #这篇文章标题背景图片
catalog: true # 是否归档
tags: #标签
  - 生信分析
---

近期拟定开展电子烟相关测试研究，但整体课题设计较为粗略，需要积累相关可研究热点，提升文章高度。烟气处理相关突变定是后续研究热点，考虑到电子烟处理可以不用进行稀释，那么细胞处理的污染性大大降低，以达到长期处理的可能性。<br>

知识主要整理于**医学统计园**公众号，仅作个人学习使用。<br>

#### Mutational Signatures 简介

人体细胞中基因的突变可分为两种: Somatic Mutation （体细胞突变）和 Germline Mutation（生殖系突变）。Germline Mutation 是指从父系或者母系（即胚胎时期）遗传得到的，可以通俗讲是“天生的突变”。**Somatic Mutation**，是指在后天发育过程中，由于 DNA 碱基之间发生置换或颠换，导致细胞基因组发生改变（这种改变并不是人体每个细胞都会携带），即“后天的突变”。

##### Mutational Signatures 概念

体细胞突变随时都会发生。诱发 DNA 突变因素很多：1. 外源因素，如化学试剂诱导，紫外线照射、**烟气处理**等；2. 内源因素，如 DNA 错配修复蛋白功能降低等。不同突变诱发过程，会在细胞内产生独特的碱基变化谱，这些碱基变化组成一类特征称之为 Mutational Signatures（可称为，突变特征）。<br>

Cosmic 数据库整理了目前较全面体细胞突变数据。其中包括在肿瘤发展不同时期，Signature 的变化特征，并认为这些特征与生物体某些特定的生物过程（代谢过程，或生物习惯）有关联。<br>

##### 不同突变特征与环境

研究发现，不同环境处理对应的突变特征也有明显的区别。<br>

1. 紫外线照射下，更容易发生 SBS7a 的突变特征，即 C>T 类颠换。
2. **吸烟更倾向于发生 SBS4 类的突变，即 C>A 类颠换。**

##### 不同突变特征与癌症

研究同样发现，不同癌种对应的突变特征也有明显的区别。<br>

1. Signature1 在所有的癌种中比较常见的一种突变类型。
2. Signature3 在乳腺癌或者卵巢癌中突变频率较高。
   Cosmic 官网上一共有 30 个 Signatures。<br>

#### Mutational Signatures 分析

了解突变特征的概念后，接着就是针对样本如何进行分析其突变特征。<br>

##### mutational signatures 类型

mutational signatures 类型主要分为三类：SBS(Single Base substitutions)单碱基替换、DBS(Double Base Substitutions)双碱基替换、IDs(Insertions and Deletions)小片段插入和缺失<br>

##### Python 环境下教程

```
# 所需环境：Python > 3.4
# 安装分析所需Python库
pip install SigProfilerMatrixGenerator

# 加载Python库，安装待分析的基因组信息
from SigProfilerMatrixGenerator import install as genInstall
genInstall.install('GRCh38',bash=False)

# Signatures分析
import os
os.chdir("D://Mutation_signature//maf") # 设置工作路径
matrices = matGen.SigProfilerMatrixGeneratorFunc("temp",
"GRCh38", "./", plot=True,exome=False,bed_file=None,
chrom_based=False,tsb_stat=False,seqInfo=False,cushion=100)

# 具体参数详解
# temp 输入的vcf文件名，还可支持maf，txt
# GRCh38 参考基因组名称
# ./ 分析结果输出为当前目录
# plot 绘制分析结果
# exome 默认False 分析全部的mutation，不只是exon
# bed_file 默认False 无bed_file文件输入
# chrom_based 默认False 不输出charom based的matrix
# seqInfo 默认False 不输出原始突变序列矩阵
# cushion 默认100 如果exome和bed_file均为False，这个参数意义也不大，是在给定位置上下游100处，统计突变

# 在当前工作路径下会产生三个文件夹：input, output, logs
```
