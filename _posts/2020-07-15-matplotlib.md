---
layout: post # 使用的布局（不需要改）
title: matplotlib # 标题
subtitle: Python中matplotlib库的学习 #副标题
date: 2020-07-15 # 时间
author: CHY # 作者
header-img: img/wallhaven-matplotlib.jpg #这篇文章标题背景图片
catalog: true # 是否归档
tags: #标签
  - 计算机
---

#### matplotlib 简介

![matplotlib绘图元素](https://github.com/chenhongyubio/chenhongyubio.github.io/raw/master/img/matplotlib绘图元素.png)<br>

1. figure<br>
   figure 是图片的载体，可以看做是画布，图片必须在画布的基础上进行创建<br>
2. axes<br>
   axes 表示绘图区域或者窗口，用来容纳一张具体的图片。axes 位于 figure 上，同一个 figure 上可以存在多个 axes。<br>
3. axis<br>
   axis 表示坐标轴，比如常见的 x 轴，y 轴以及对应的刻度线和标签。<br>
4. artist<br>
   atrist 表示广义的绘图元件，往大了说，figure, axes, axis 都属于绘图元件，往小了说，图片标题，图例，刻度，刻度标签等一幅图像上的具体元素都属于绘图元件。<br>

```
# 通过plt.subplots创建一个figure和axes对象，然后通过axes对象的各种方法来创建图表
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], [1, 2, 3, 4])
```

##### 绘图输出结果 backend

backend 可以分为两类：

1. 交互式设备，比如 qt,wxpython 等 GUI 窗口
   ![matplotlib输出1](https://github.com/chenhongyubio/chenhongyubio.github.io/raw/master/img/matplotlib输出1.png)<br>
2. 非交互式设备，比如 png, jpeg, pdf 等各种格式的文件
   ![matplotlib输出2](https://github.com/chenhongyubio/chenhongyubio.github.io/raw/master/img/matplotlib输出2.png)<br>

##### 修改内置 backend

```
# 修改配置文件matplotlibrc
import matplotlib
matplotlib.matplotlib_fname() # 找到配置文件位置
 # 修改backend值

# 设置环境变量
export  MPLBACKEND='agg'

# 绘图脚本的开头进行设置
import matplotlib
matplotlib.use('agg')
```

```
# 弹出对应结果
plt.show()  # 交互式图形设备
plt.savefig('out.png') # 非交互式图形设备
```

#### matplotlib 基础绘图命令之 plot

```
# plot命令主要用于绘制折线图
import matplotlib.pyplot ad plt
plt.plot([1,2,3,4],[1,2,3,4])
# 第一个参数的值作为x轴坐标，第二个参数的值作为y轴坐标
# 当只提供一个数值参数时，自动将其作为y轴坐标，x轴坐标为对应的数值下标
import pandas as pd
data_dict = pd.DataFrame({'xlabel':[1, 2, 3, 4], 'ylabel':[1, 2, 3, 4]})
plt.plot('xlabel', 'ylabel', data=data_dict)
```

```
# plot绘制散点图
import numpy as np
x = np.array([1, 2, 3, 4])
y = np.array([1, 2, 3, 4])
plt.plot(x,y,'o')  # 散点图
plt.plot(x,y,marker='o', linestyle='--', linewidth=2) # 散点图与直线图叠加
```

对于点而言，拥有以下基本属性

1. 填充色， markerfillcolor, 简写为 mec
2. 边框颜色，markeredgecolor, 简写为 mfc
3. 边框的线条宽度，markeredgewidth, 简写为 mfc
4. 大小, markersize, 简写为 ms
5. 形状, marker

对于线而言，用于以下基本属性

1. 颜色, color, 简写为 c
2. 宽度，linewidth, 简写为 lw
3. 风格，实线还是虚线，linestyle. 简写为 ls

![plot参数简写1](https://github.com/chenhongyubio/chenhongyubio.github.io/raw/master/img/plot参数简写1.png)<br>
![plot参数简写2](https://github.com/chenhongyubio/chenhongyubio.github.io/raw/master/img/plot参数简写2.png)<br>
![plot参数简写3](https://github.com/chenhongyubio/chenhongyubio.github.io/raw/master/img/plot参数简写3.png)<br>

```
# 图形叠加
plt.plot(x, y, label = 'sampleA')
plt.plot(x, y + 1, label = 'sampleB')
plt.plot(x, y + 2, label = 'sampleC')
plt.legend()
```

#### matplotlib 基础绘图命令之 bar

bar 命令用于绘制柱状图。

```
plt.bar(x = [1, 2, 3, 4], height = [4, 2, 3, 1])
```

参数 x 的值作为 x 轴坐标，height 的值作为 y 轴坐标。<br>

1. width, 柱子的宽度，即在 x 轴上的长度，默认是 0.8
2. color, 柱子的填充色
3. edgecolor, 柱子边框的颜色，默认为 None
4. linewidth, 柱子边框的宽度，默认为 0，表示没有边框
5. yerr,指定误差值的大小， 用于在柱子上添加误差线
6. ecolor, 表示 errorbar color, 误差线的颜色
7. bottom, 柱子底部的 baseline, 默认为 0

```
# 堆积柱状图
plt.bar(x = [1, 2, 3, 4], height = [4, 3, 2, 1], label = 'sampleA')
plt.bar(x = [1, 2, 3, 4], height = [4, 3, 2, 1], bottom = [4, 3, 2, 1], label = 'sampleB')
plt.legend()
# 核心是通过将第一组柱子的高度作为第二组柱子的底部，即bottom参数，从而实现堆积的效果
```

```
# 分组柱状图
width = 0.4
plt.bar(x = np.array([1, 2, 3, 4]) - width / 2, height = [4, 3, 2, 1], width = width, label = 'sampleA')
plt.bar(x = np.array([1, 2, 3, 4]) + width / 2, height = [1, 2, 3, 4], width = width, label = 'sampleB')
plt.legend()
# 核心是根据宽度的值，手动计算柱子的中心坐标，然后自然叠加就可以形成水平展开的分组柱状图
```

#### matplotlib 基础绘图命令之 pie

matplotlib 中，pie 方法用于绘制饼图。

```
plt.pie(x=[1, 2, 3, 4])
```

pie 方法常用的参数:<br>

1. labels, 设置饼图中每部分的标签<br>
2. autopct, 设置百分比信息的字符串格式化方式，默认值为 None,不显示百分比<br>
3. shadow, 设置饼图的阴影，使得看上去有立体感，默认值为 False<br>
4. startangle, 饼图中第一个部分的起始角度，<br>
5. radius, 饼图的半径，数值越大，饼图越大<br>
6. counterclock, 设置饼图的方向，默认为 True,表示逆时针方向，值为 False 时为顺时针方向<br>
7. colors，调色盘，默认值为 None, 会使用默认的调色盘，所以通常情况下，不需要设置该参数<br>
8. explode, 该参数用于突出显示饼图中的指定部分<br>

```
# 为了将图例和内容有效的区分开来，可以通过设置legend方法的bbox_to_anchor参数,
# 该参数用于设置图例区域在figure上的坐标，其值为4个元素的元组，分别表示x,y,width,height,
data=[1,2,3,4]
labels=['sampleA', 'sampleB', 'sampleC', 'sampleD']
plt.pie(x=data, labels=labels, autopct=lambda pct:'({:.1f}%)\n{:d}'.format(pct, int(pct/100 * sum(data))))
plt.legend(labels,loc="upper left",bbox_to_anchor=(1.2, 0, 0.5, 1))
```

#### matplotlib 基础绘图命令之 scatter

scatter 方法用于绘制散点图，与 plot 方法不同之处在于，scatter 主要用于绘制点的颜色和大小呈现梯度变化的散点图，也就是我们常说的气泡图。<br>

```
plt.scatter(x= np.random.randn(10), y=np.random.randn(10),s=40 * np.arange(10),c=np.random.randn(10))
# x和y参数指定x轴和y轴坐标，s参数指定mark size, 即点的大小，c参数指定color,即颜色。scatter会根据数值自动进行映射
# 难点在于其图例的处理上。scatter函数的返回值为一个PathCollections对象，通过其legend_elements方法，可以获得绘制图例所需的信息

scatter = plt.scatter(x= np.random.randn(10), y=np.random.randn(10),s=40 * np.arange(10),c=np.random.choice(np.arange(4), 10))
plt.legend(*scatter.legend_elements())

scatter = plt.scatter(x= np.random.randn(10), y=np.random.randn(10),s=40 * np.arange(10),c=np.random.choice(np.arange(4), 10))
plt.colorbar(scatter)

# legend_elements方法是有很多参数可以调整的，其中prop参数指定返回的信息，有两种取值，默认是colors, 表示返回的是点的颜色信息，取值为sizes时，返回的是点的大小信息。另外还有一个参数是num, 当图例的取值为连续型时，num指定了图例上展示的点的个数
scatter = plt.scatter(x= np.random.randn(10), y=np.random.randn(10),s=40 * np.arange(10),c=np.random.choice(np.arange(4), 10))
plt.legend(*scatter.legend_elements(prop='sizes', num = 6))

# 组合图例
fig, ax = plt.subplots()
scatter = ax.scatter(x= np.random.randn(10), y=np.random.randn(10),s=40 * np.arange(10),c=np.random.choice(np.arange(4), 10))
legend1 = ax.legend(*scatter.legend_elements(prop='colors'), loc='upper left',title='colors',bbox_to_anchor=(1, 0, 0.5, 1))
ax.add_artist(legend1)
legend2 = ax.legend(*scatter.legend_elements(prop='sizes', num = 6), loc='lower left',title='sizes',bbox_to_anchor=(1, 0, 0.5, 1))
```

#### matplotlib 基础绘图命令之 errorbar

在 matplotlib 中，errorbar 方法用于绘制带误差线的折线图。<br>

```
plt.errorbar(x=[1, 2, 3, 4], y=[1, 2, 3, 4], yerr=1)
# yerr参数用于指定y轴水平的误差，同时该方法也支持x轴水平的误差，对应参数xerr。指定误差值有多种方式，上述代码展示的是指定一个统一标量的用法，此时，所以的点误差值都一样。
plt.errorbar(x=[1, 2, 3, 4], y=[1, 2, 3, 4], yerr=[1, 2, 3, 4]) # 不同误差线
plt.errorbar(x=[1, 2, 3, 4], y=[1, 2, 3, 4], yerr=[[1,2,3,4],[1, 2, 3, 4]]) # 每个点上下误差不一样
```

```
# 样式更改
# fmt参数的值和plot方法中指定点的颜色，形状，线条风格的缩写方式相同
plt.errorbar(x=[1, 2, 3, 4], y=[1, 2, 3, 4], yerr=1, fmt='co--')
# ecolor参数指定error bar的颜色，可以和折线的颜色加以区分
plt.errorbar(x=[1, 2, 3, 4], y=[1, 2, 3, 4], yerr=1, fmt='co--', ecolor='g')
# elinewidth参数指定error bar的线条宽度
plt.errorbar(x=[1, 2, 3, 4], y=[1, 2, 3, 4], yerr=1, fmt='ro-',ecolor='k',elinewidth=10)

# lims系列参数
# lims系列参数用于控制误差线的显示，对于x轴水平的误差线而言，有以下两个参数
# 1. xuplims
# 2. xlolims
# 对于y轴水平的误差线而言，有以下两个参数
# 1. uplims
# 2. lolims
# 默认的取值为False， 当取值为True时，对应方向的误差线不显示，同时在另外一个方向上的误差线上，会用箭头加以标识。
plt.errorbar(x=[1, 2, 3, 4], y=[1, 2, 3, 4], yerr=1, uplims=True)
plt.errorbar(x=[1, 2, 3, 4], y=[1, 2, 3, 4], yerr=1, lolims=True)
plt.errorbar(x=[1, 2, 3, 4], y=[1, 2, 3, 4], yerr=1, uplims=True, lolims=True)

# errorevery参数用于指定误差线的抽样频率，默认情况下，每个点的误差线都会显示，当点很多且密集分布时, 每个点都显示误差线的话，就很难看出有效的信息，
plt.errorbar(x=range(100), y=range(100),yerr=50)
plt.errorbar(x=range(100), y=range(100),yerr=50,errorevery=6) # 使用errorevery参数进行抽样

# 样式精细调整
plt.errorbar(x=[1, 2, 3, 4], y=[1, 2, 3, 4], yerr=1, marker='s', mfc='red', mec='green', ms=20, mew=4)
```

#### 一文搞懂 matplotlib 中的颜色设置

##### 常用颜色的字母表示及缩写

```
1. red,表示红色,  简写为r
2. green, 表示绿色，简写为g
3. blue,表示蓝色，简写为b
4. yellow,表示黄色，简写为y
5. cyan,表示蓝绿色，简写为c
6. magenta,表示粉紫色，简写为m
7. black,表示黑色，简写为k
8. white,表示白色，简写为w
```

##### T10 调色盘

在 matplotlib 中，默认的颜色盘通过**参数 rcParams["axes.prop_cycle"]参数**来指定, 初始的调色盘就是 T10 调色盘。<br>
T10 调色盘适用于离散分类，其颜色名称以 tab:为前缀。<br>
在 matplotlib 中，默认就是通过这个 T10 调色盘来个不同的 label 上色的。<br>

```
1. tab:blue
2. tab:orange
3. tab:green
4. tab:red
5. tab:purple
6. tab:brown
7. tab:pink
8. tab:gray
9. tab:olive
10. tab:cyan
```

##### CN 式写法

CN 式写法以字母 C 为前缀，后面加从 0 开始的数字索引，其索引的对象为 rcParams["axes.prop_cycle"]指定的调色盘。CN 式对应调色板，当切换调色板时，CN 也会对应改变<br>

##### xkcd 颜色名称

在 matplotlib 中，通过 xkcd:前缀加对应的颜色名称进行使用，而且是不区分大小写的。具体网站为：https://xkcd.com/color/rgb/ <br>

##### X11/CSS4 颜色名称

```
# 通过字典查看颜色
import matplotlib._color_data as mcd
for key in mcd.CSS4_COLORS:
    print('{}: {}'.format(key, mcd.CSS4_COLORS[key]))
```

##### 十六进制颜色代码

```
# 十六进制的颜色代码可以精确的指定颜色，在matplotlib中也支持
plt.pie(x=[1,2,3,4], colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
```

##### RGB/RGBA 元组

```
# 所有的颜色都是有RGB三原色构成，在matplotlib中，可以通过一个元组来表示表示red, green, blue三原色的比例，以及一个可选的alpha值来表示透明度，取值范围都是0到1
plt.pie(x=[1,2,3,4], colors=[(0.1, 0.2, 0.5),(0.1, 0.3, 0.5),(0.1, 0.4, 0.5),(0.1, 0.5, 0.5)])
```

##### 灰度颜色

```
# 在matplotlib中，通过0到1之间的浮点数来对应灰度梯度，在使用时，为了有效区分，需要通过引号将其装换为字符
plt.pie(x=[1,2,3,4], colors=['0','0.25', '0.5', '0.75'])
```
