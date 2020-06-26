---
layout:     post   				    # 使用的布局（不需要改）
title:      StackedVlnPlot				# 标题 
subtitle:   R语言实现单细胞StackedVlnPlot图 #副标题
date:       2020-06-26				# 时间
author:     CHY					# 作者
header-img: img/wallhaven-2069.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - 单细胞
---

小提琴堆叠图是单细胞分析中常用于展示Marker gene表达的方式，简单直观，这里记录一下几种绘制小提琴堆叠图的方法。

#### R原生函数实现StackedVlnPlot
```
library(Seurat)
library(ggplot2)
# 构建函数
modify_vlnplot<- function(obj,
                          feature,
                          pt.size = 0,
                          plot.margin = unit(c(-0.75, 0, -0.75, 0), "cm"),
                          ...) {
  p<- VlnPlot(obj, features = feature, pt.size = pt.size, ... )  +
    xlab("") + ylab(feature) + ggtitle("") +
    theme(legend.position = "none",
          axis.text.x = element_blank(),
          axis.text.y = element_blank(),
          axis.ticks.x = element_blank(),
          axis.ticks.y = element_line(),
          axis.title.y = element_text(size = rel(1), angle = 0, vjust = 0.5),
          plot.margin = plot.margin )
  return(p)
}
StackedVlnPlot<- function(obj, features,
                          pt.size = 0,
                          plot.margin = unit(c(-0.75, 0, -0.75, 0), "cm"),
                          ...) {
  plot_list<- purrr::map(features, function(x) modify_vlnplot(obj = obj,feature = x, ...))
  plot_list[[length(plot_list)]]<- plot_list[[length(plot_list)]] +
    theme(axis.text.x=element_text(), axis.ticks.x = element_line())
  
  p<- patchwork::wrap_plots(plotlist = plot_list, ncol = 1)
  return(p)
}
# 颜色设置
my36colors <-c('#E5D2DD', '#53A85F', '#F1BB72', '#F3B1A0', '#D6E7A3', '#57C3F3', '#476D87',
               '#E95C59', '#E59CC4', '#AB3282', '#23452F', '#BD956A', '#8C549C', '#585658',
               '#9FA3A8', '#E0D4CA', '#5F3D69', '#C5DEBA', '#58A4C3', '#E4C755', '#F7F398',
               '#AA9A59', '#E63863', '#E39A35', '#C1E6F3', '#6778AE', '#91D0BE', '#B53E2B',
               '#712820', '#DCC1DD', '#CCE0F5',  '#CCC9E6', '#625D9E', '#68A180', '#3A6963',
               '#968175'
)
# 函数运行
StackedVlnPlot(sdata, c('Retnlg', 'Pygl', 'Anxa1', 'Igf1r', 'Stfa2l1'), pt.size=0, cols=my36colors)
```