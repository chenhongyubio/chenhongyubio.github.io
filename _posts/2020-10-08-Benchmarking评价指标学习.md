---
layout: post # 使用的布局（不需要改）
title: 多种评价指标 # 标题
subtitle: Benchmarking评价指标学习 #副标题
date: 2020-10-08 # 时间
author: CHY # 作者
header-img: img/wallhaven-j5mr6w.jpg #这篇文章标题背景图片
catalog: true # 是否归档
tags: #标签
  - 单细胞
---

本章主要记录学习多种评价指标原理、应用场景及实现方法的笔记，用于后续 Benchmarking 分析研究。<br>

### 定量指标 Quantitative

#### 与基本事实比较 Comparison to ground truth

##### 结果离散 Discrete target

主要需要基于混淆矩阵进行后续计算。<br>
True Positive(TP)：将正类预测为正类 <br>
True Negative(TN)：将负类预测为负类 <br>
False Positive(FP)：将负类预测为正类数 → 误报 (Type I error) <br>
False Negative(FN)：将正类预测为负类数 → 漏报 (Type II error) <br>

**True positive rate (TPR)**
真阳率：检测出来的**真阳性样本数**除以所有**真实阳性样本数**<br>
TPR = TP / (TP + FN)<br>

**False positive rate (FPR)**
假阳率：检测出来的**假阳性样本数**除以所有**真实阴性样本数**<br>
FPR = FP / (TN + FP)<br>

**False discovery rate (FDR)**
误报率：所有发现中发生错误所占的比率<br>
FDR = FP / (FP + TP)<br>

**Sensitivity and specificity**
Sensitivity：方法对阳性的敏感度，对于一堆金标准的阳性结果，你能判断出多少来。<br>
specificity：方法能指定地判断出阴性的能力，对于一堆金标准的阴性结果，你会不会误判出一些阳性的结果。<br>

**precision and recall**
精确率(查全率)：TP / (TP + FP)<br>
召回率(查准率)：TP / (TP + FN)<br>

**F1 score**
当精确率和召回率不能很好比较两者时，采用 F1-score 来进行判断。<br>
F1 分数可以看作是模型精确率和召回率的一种调和平均，它的最大值是 1，最小值是 0。<br>
F1 = 2 _ ((precision _ recall) / (precision + recall))<br>

**Adjusted Rand index**
Rand index：(TP + TN) / (TP + TN + FP + FN) RI 存在缺点，就是惩罚力度不够，换句话说，大家普遍得分比较高，没什么区分度，遍地 80 分，区分不明显。<br>
ARI 相对于 RI 来说更加具有区分度。<br>
ARI = (RI - E[RI]) / (max(RI) - E[RI]) 取值为[-1,1] <br>

**Normalized mutual information**
归一化互信息：常用在聚类中，度量 2 个聚类结果的相近程度 <br>

**ROC / TPR-FDR / PR curves**
ROC：假阳率当 x 轴，真阳率当 y 轴画一个二维平面直角坐标系 <br>

**Area under ROC / PR curves**
AUC：ROC 下的面积

**Classification accuracy (supervised)**
ACC：描述分类器的分类准确率
ACC = (TP + TN) / (TP + FP + FN + TN)

##### 结果连续 Continuous target

**Root mean square error**

**Distance measures**
**Pearson correlation**
**Sum of absolute log-ratios**
**Log-modulus**
**Cross-entropy**
**Prediction accuracy (supervised)**

##### 复杂结果 Complex target

**Graph similarity**
**Tree similarity**
**Distribution similarity**
**Custom metrics**

#### 无基本事实比较 No ground truth

**Stability**
**Stochasticity**
**Robustness**
**Null comparisons**
**Runtime**
**Scalability**
**Memory requirements**

### 定性指标 Qualitative

**User-friendliness**
**Documentation quality**
**Installation procedures**
**Freely available / open source**
**Code quality**
**Use of unit testing**
**Use of continuous integration**

[机器学习模型评估指标总结！](https://mp.weixin.qq.com/s/FLMbkU70EZj5f4ISjDDIVw)<br>
[一个自动计算各种检验指标的工具](https://mp.weixin.qq.com/s/sjTSBsQXaPkqD2P3R1-t9Q)<br>