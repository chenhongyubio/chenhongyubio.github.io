---
layout: post # 使用的布局（不需要改）
title: bamtofastq使用 # 标题
subtitle: 10X单细胞数据如何从Bam转回Fastq #副标题
date: 2021-08-16 # 时间
author: CHY # 作者
header-img: img/wallhaven-l3mx32.jpg #这篇文章标题背景图片
catalog: true # 是否归档
tags: #标签
  - 单细胞
---
详细链接：https://support.10xgenomics.com/docs/bamtofastq <br>

bamtofastq是10X官方专门开发的针对cellranger, cellranger-atac, cellranger-arc, cellranger-dna, spaceranger or longranger等软件生成的Bam倒推出fastq文件的工具。BAMs produced for TCR or BCR data, by aligning to a V(D)J reference with cellranger vdj or cellranger multi, are not supported by bamtofastq. Special tags是必须的，不然会报“WARNING: no @RG (read group) headers found in BAM file.”错误。所以从SRA下载的BAM文件也需要注意是否去除Tag。<br>

```
/home/chenhy/software/bamtofastq-1.3.2 --nthreads=8 shr_root_mut_postsorted_genome_bam.bam
```