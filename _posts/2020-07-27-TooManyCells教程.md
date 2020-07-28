---
layout: post # 使用的布局（不需要改）
title: TooManyCells教程 # 标题
subtitle: scRNA-seq数据分析绘图 #副标题
date: 2020-07-27 # 时间
author: CHY # 作者
header-img: img/wallhaven-toomanycells.png #这篇文章标题背景图片
catalog: true # 是否归档
tags: #标签
  - 单细胞
---

TooManyCells 算法有 2 种包装形式，一种是 too-many-cells 软件，一种是 TooManyCellsR R 包。<br>

#### TooManyCells 方法原理

TooManyCells 通过递归技术反复识别在树结构中亚群，其可视化模型结合一系列可视化特性为细胞状态提供一个灵活的平台，跟踪、探索和检测稀有种群。除了聚类和可视化之外，TooManyCells 还提供其他功能，包括但不限于异质性评估、聚类测量、多样性和稀有性统计。TooManyCells 同时识别稀有和丰富细胞群体的优越性可在三个独立的环境中得到证明，在控制环境中，该方法不仅将两个稀有细胞群体从普通细胞群体中分离出来，而且成功地将两个稀有细胞群体分离。

1. **聚类**：TooManyCells 实现了最初用于文本挖掘的无矩阵分层谱聚类，使用该聚类方法的最终结果是一种树状结构，其中每个内部节点是一个粗略的簇，每个叶子是每个模块化度量中最精细的簇。
2. **可视化**：TooManyCells 算法使用 BirchBeer 渲染方法显示单细胞群集层次结构。
3. **差异表达**：给定多个群集标识号，TooManyCells 可以执行差异表达分析以识别这些群集中细胞的基因表达之间的差异。

#### TooManyCells 软件版使用流程

```
# 安装
# 第一步：conda 安装 curl
conda install curl
# 第二步：用 curl 安装 stack
curl -sSL https://get.haskellstack.org/ | sh
stack setup
# 第三步：用 stack 安装 too-many-cells
git clone https://github.com/GregorySchwartz/too-many-cells.git
cd too-many-cells
stack install
# 完成后打开nohup.out 看到最后提示将/blabla/.local/bin 路径加入PATH变量
```

```
# 输入既可以是一个文件夹（里面放 cellranger 的 3 个文件），也可以是一个 csv 格式的普通表达矩阵
# 除了表达矩阵之外，还需要一个输入文件labels.csv
too-many-cells make-tree \
        --matrix-path ../input/expr_count.csv \
        --labels-file ../input/labels.csv \
        --draw-collection "PieRing" \
        --output ../out \
        > ../out/clusters.csv

# 修剪树枝
# 直接设置 --min-size 参数为一个值，如 100，以规定最小分支细胞数
# 设置 --smart-cutoff 参数为一个值，如 4，以规定最小分支细胞数为 4*median absolute deviation
too-many-cells make-tree \
    --prior ../out \
    --labels-file ../input/labels.csv \
    --smart-cutoff 1 \ #经调试，我的数据最合适的值是1
    --min-size 1 \
    --draw-collection "PieChart" \
    --output ../out_pruned \
    > ../out_pruned/clusters_pruned.csv

# 标记数字
too-many-cells make-tree \
    --prior ../out \
    --labels-file ../input/labels.csv \
    --smart-cutoff 1 \
    --min-size 1 \
    --draw-collection "PieChart" \
    --draw-node-number \ #只需多加这个参数
    --output ../out_pruned \
    > ../out_pruned/clusters_pruned.csv

#  分支末端画成饼图
too-many-cells make-tree \
    --prior out \
    --labels-file labels.csv \
    --smart-cutoff 4 \
    --min-size 1 \
    --draw-collection "PieChart" \
    --output out_pruned \
    > clusters_pruned.csv

# 调整树枝宽度
too-many-cells make-tree \
    --prior out \
    --labels-file labels.csv \
    --smart-cutoff 4 \
    --min-size 1 \
    --draw-collection "PieChart" \
    --draw-max-node-size 40 \
    --output out_pruned \
    > clusters_pruned.csv

# 不按分支大小缩放
too-many-cells make-tree \
    --prior out \
    --labels-file labels.csv \
    --smart-cutoff 4 \
    --min-size 1 \
    --draw-collection "PieChart" \
    --draw-max-node-size 40 \
    --draw-no-scale-nodes \
    --output out_pruned \
    > clusters_pruned.csv

# 特定基因表达量
too-many-cells make-tree \
    --prior ../out \
    --matrix-path ../input/expr_count.csv \
    --labels-file ../input/labels.csv \
    --smart-cutoff 1 \
    --min-size 1 \
    --feature-column 2 \
    --draw-leaf "DrawItem (DrawThresholdContinuous [(\"gene1\", 0), (\"gene2\", 0)])" \
    --draw-colors "[\"#e41a1c\", \"#377eb8\", \"#4daf4a\", \"#eaeaea\"]" \
    --draw-scale-saturation 10 \ #如果不加这个参数，很可能表达量普遍较低以至于整张图没有颜色，至于这个值多少比较合适我还没有试过
    --output ../out_gene_expression \
    > ../out_gene_expression/clusters_pruned.csv

# 任意分支之间的差异分析
too-many-cells differential \
    --matrix-path ../input/expr_count.csv \
    -n "([110], [148])" \
    +RTS -N24
    > ../out/differential.csv

# 比较两个细胞群的多样性
too-many-cells diversity\
    --priors ../out1 \
    --priors ../out2 \
    -o ../out_diversity_stats

# 伪时序分析
too-many-cells paths\
    --prior ../out \
    --labels-file ../input/labels.csv \
    --bandwidth 3 \
    -o ../out_paths
```

#### TooManyCellsR R 版本使用教程

仅仅对软件版进行了 R 封装。

```
tooManyCells(mat, args = c("make-tree", "--smart-cutoff", "4", "--min-size", "1"))
plot(res$treePlot, axes = FALSE)
res$stdout
res$nodeInfo
plot(res$clumpinessPlot, axes = FALSE)
```

#### 参考文档

https://github.com/GregorySchwartz/tooManyCellsR?tdsourcetag=s_pctim_aiomsg <br>
https://github.com/GregorySchwartz/too-many-cells <br>
