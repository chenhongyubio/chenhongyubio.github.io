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
