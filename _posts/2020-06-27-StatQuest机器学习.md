---
layout:     post   				    # 使用的布局（不需要改）
title:      机器学习				# 标题 
subtitle:   StatQuest机器学习总结 #副标题
date:       2020-06-27			# 时间
author:     CHY					# 作者
header-img: img/wallhaven-2064.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - 机器学习
---
本节主要记录StatQuest中机器学习部分的内容。

#### 机器学习基础简介
一般来讲，机器学习就是进行预测和分类。<br>
通过训练数据集训练模型，然后使用测试数据集验证模型的好坏。<br>

#### 交叉验证
交叉验证使我们能够比较不同的机器学习方法并对这些方法在实践中的表现有所了解。<br>
1. 为机器学习方法估计参数（训练算法）
2. 评价选择的方法是否能够很好地对新数据进行分类（预测）（测试算法）<br>
一般是将前75%的数据用于训练算法，后25%用于测试算法。<br>
交叉验证会选择不同位置的数据进行训练，然后比较预测效果。（如四重交叉验证、十折交叉验证）

#### 混淆矩阵The Confusion Matrix
![简单混淆矩阵](https://github.com/chenhongyubio/chenhongyubio.github.io/raw/master/img/简单混淆矩阵.png)
![复杂混淆矩阵](https://github.com/chenhongyubio/chenhongyubio.github.io/raw/master/img/复杂混淆矩阵.png)
绿色代表预测正确，红色代表预测错误。<br>
根本上来看，混淆矩阵的大小取决于想要预测的事物的数量。<br>

#### 敏感性和特异性
##### 2x2混淆矩阵中的计算
敏感性Sensitivity: 正确识别出阳性结果的结果百分比
![敏感性](https://github.com/chenhongyubio/chenhongyubio.github.io/raw/master/img/Sensitivity.png)
特异性Specificity: 正确识别出阴性结果的结果百分比
![特异性](https://github.com/chenhongyubio/chenhongyubio.github.io/raw/master/img/Specificity.png)
3x3混淆矩阵中的计算与2x2的类似，其中Specificity不太一样，是其余所有的整合为一个进行计算。<br>
敏感性和特异性可以用于选择哪种方法更加适合我们的数据。

#### 偏差和方差
用来评估模型是否很好匹配数据，用于后续预测。
The inability for a machine learning method to capture the true relationship is called bias.<br>
In Machine Learning lingo,the difference in fits between data sets is called Variance.<br>
Lower bias,Lower variance,more accurate.

#### ROC and AUC
纵坐标：True Positive Rate,which is the same thing as Sensitivity.
横坐标：False Positive Rate, which is the same thing as 1 - Specificity.
经常使用Precision 替代 False Positive Rate.
![准确度](https://github.com/chenhongyubio/chenhongyubio.github.io/raw/master/Precision.png)

#### ROC and AUC in R
```
library(pROC)
library(randomForest)
set.seed(123)
num_sample <- 100
weight <- sort(rnorm(n=num.samples,mean=172,sd=29))
obese <- ifelse(test=(rubif(n=num.samples) < (rank(weight)/100)),yes=1,no=0)
obese
plot(x=weight,y=obese)
glm.fit=glm(obese ~ weight,family = binmial)
lines(weight,glm.fit$fitted.values)
roc(obese,glm.fit$fitted.values,plot=TRUE,legacy.axes=TRUE,percent=TRUE,xlab="False Positive Percentage", ylab="True Postive Percentage",col = "#377eb8",lwd=4,)
par(pty = "s")
roc.info <- roc(obese,glm.fit$fitted.values,plot=TRUE,legacy.axes=TRUE)
roc.df <- data.frame(tpp=roc.info$sensitivities*100,fpp=(1-roc.info$specifivities)*100, thresholds=roc.info$thresholds)
head(roc.df)
roc.df[roc.df$tpp > 60 & roc.df$tpp < 80,]
```

#### Fitting a line to data最小二乘法
点到线的垂直距离平方和(残差平方和)最小。<br>
通过取导数来找到等于0的位置来找最小点。

#### 比率和比率对数 Odds and log Odds
Odds are not probabilities比率不是概率。<br>
Odds = 事情发生和事件不发生的比 <br>
Probability = 事件发生和所有可能发生的事件的比 <br>
log Odds更为对称。<br>

#### 比率比和比率比对数 Odds Ratios
比率比指的是两个比率的比。<br>
比率比可以用来判断两个事物之间是否有强或弱的关系：如果某个人有突变基因，是否比率越高对应患癌症的几率更高？(类似的问题)<br>
##### 3种方法判断比率比是否存在统计学意义
1. Fisher's Exact Test
2. Chi-Square Test卡方检验
3. The Wald Test(通常用于确定逻辑回归中比率比的显著性并计算置信区间)

#### 逻辑回归
Logistic regression predicts whether something is True or False,instead of predicting something continuous like size.<br>
逻辑回归可以利用离散变量和连续性变量。

#### 逻辑回归详解之系数Coefficients
Coefficients are the result of any Logistic Regression.<br>
Generalized Linear Model广义线性模型。<br>
逻辑回归中的Y轴：0~1的概率值。可以**转换为log(odds)**<br>
X轴可以为连续变量，也可以为离散变量。<br>
Coefficients中第一个参数Estimate相当于直线与Y轴的截距，Std. Error针对Estimate的标准误差，第二行的参数就是斜率。<br>
![两种变量](https://github.com/chenhongyubio/chenhongyubio.github.io/raw/master/img/逻辑回归Coefficients.png)

#### 极大似然maximum likelihood
极大似然用于判断逻辑回归中线条是否很好的匹配数据。
1. 在逻辑回归中讲Probability转换为以log(odds)作为Y轴
2. (转换后原始数据存在无穷的位置无法计算距离平方和) 将原始点映射到以log(odds)为Y轴构建的直线中，计算出每个原始点的log(odds)值
3. 通过公式将得到的log(odds)值转换为probabilities值
4. 将得到的每个点的probabilities值log转换后相乘

![两种变量](https://github.com/chenhongyubio/chenhongyubio.github.io/raw/master/img/逻辑回归--极大似然1.png)

#### 逻辑回归中的R2 and p-values
存在多种方法来计算，并没有得出一个共识。<br>
McFadden's Pseudo R2一种常用且简单的计算方法。<br>
线性模型中R2越大，拟合效果越好。<br>
![线性模型R2计算.png](https://github.com/chenhongyubio/chenhongyubio.github.io/raw/master/img/线性模型R2计算.png)
![](https://github.com/chenhongyubio/chenhongyubio.github.io/raw/master/img/宋.png)

#### 逻辑回归R实现
```
# 数据处理
url <- "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
data <- read.csv(url,header = FALSE)
head(data)
colnames(data) <- c("age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang",'oldpeak","slope","ca","thal","hd")
head(data)
str(data)
data[data == "?"] <- NA
data[data$sex == 0,]$sex <- "F"
data[data$sex == 1,]$sex <- "M"
data$sex <-as.factor(data$sex)
data$cp <-as.factor(data$cp)
data$fbs <-as.factor(data$fbs)
data$restecg <-as.factor(data$restecg)
data$exang <-as.factor(data$exang)
data$slope <-as.factor(data$slope)
data$ca <-as.integer(data$ca) 
data$ca <-as.factor(data$ca)
data$thal <-as.integer(data$thal)
data$thal <-as.factor(data$thal)
data$hd <-ifelse(test=data$hd == 0, yes= "Healthy", no="Unhealthy")
data$hd <-as.factor(data$hd)
str(data)
nrow(data[is.na(data$ca) | is.na(data$tha),])
data[is.na(data$ca) | is.na(data$thal),]
data <- data[!(is.na(data$ca) | is.na(data$thal)),]
nrow(data)
xtabs(~ hd + sex,data = data)
xtabs(~ hd + cp,data = data)
xtabs(~ hd + fbs,data = data)
# 逻辑回归
logistic <- glm(hd ~ sex,data = data,family="binomial")
summary(logistic)
logistic <- glm(hd ~ ,data = data,family="binomial")
ll.null <- logistic$null.deviance/-2
ll.proposed <- logistic$deviance/-2
(ll.null - ll.proposed) / ll.null
1- pchisq(2*(ll.proposed - ll.null),df=(length(logistic$coefficients)-1))
# 绘图
predicted.data <- data.frame(probability.of.hd = logistic$fitted.values,hd=data$hd)
predicted.data <- predicted.data[order(predicted.data$predicted.of.hd,decreasing=FALSE),]
predicted.data$rank <- 1:nrow(predicted.data)
library(ggplot2)
library(cowplot)
ggplot(data-predicted.data, aes(x=rank, y=probability.of.hd)) +
    geom.point(aes(color=hd), alpha=1, shape=4, stroke=2) +
    xlab("Index") +
    ylab("Predicted probability of getting heart disease")
```